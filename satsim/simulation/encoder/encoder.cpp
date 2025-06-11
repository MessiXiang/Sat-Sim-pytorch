#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>
#include <cmath>

#include "encoder.h"

extern "C" {
    PyObject* PyInit__C(void) {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_C",
            "This module is a encoder kernel register trigger.\n\nBefore import this, make sure you have imported torch first. You should never use this module.",
            -1,
            NULL,
        };
        return PyModule_Create(&module_def);
    }
}


namespace encoder {
    std::tuple<torch::Tensor, torch::Tensor> forward_cpu(
        torch::Tensor target_speeds,
        torch::Tensor remaining_clicks,
        torch::Tensor signals,
        torch::Tensor speeds,
        double clicks_per_radian,
        double dt
    ) {
        target_speeds = target_speeds.contiguous();
        remaining_clicks = remaining_clicks.contiguous();
        signals = signals.contiguous();
        speeds = speeds.contiguous();
        
        speeds = speeds.clone();

        torch::Tensor mask;
    
        mask = (signals == int(ReactionWheelSignal::NOMINAL));
        if (mask.any().item<bool>()) {
            torch::Tensor target_radian = target_speeds * dt;
            torch::Tensor target_clicks = target_radian * clicks_per_radian + remaining_clicks;
            
            remaining_clicks = torch::where(
                mask,
                torch::fmod(target_clicks, 1.0),
                remaining_clicks
            );
            
            speeds = torch::where(
                mask,
                torch::floor(target_clicks) / (dt * clicks_per_radian),
                speeds
            );
        }
        
        mask = (signals == int(ReactionWheelSignal::STOPPED));
        if (mask.any().item<bool>()) {
            remaining_clicks = torch::where(
                mask,
                0.,
                remaining_clicks
            );
            
            speeds = torch::where(
                mask,
                0.,
                speeds
            );
        }
        
        if (torch::all(signals == int(ReactionWheelSignal::LOCKED)).item<bool>()) {
            remaining_clicks = remaining_clicks.clone();
        }
        
        return {speeds, remaining_clicks};

    }
    TORCH_LIBRARY(encoder, m) {
        m.def("c(Tensor target_speeds, Tensor remaining_clicks, Tensor signals, Tensor speeds, float clicks_per_radian, float dt) -> (Tensor, Tensor)");
    }
    
    TORCH_LIBRARY_IMPL(encoder, CPU, m) {
        m.impl("c", &forward_cpu);
    }
}

