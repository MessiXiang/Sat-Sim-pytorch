#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>
#include <cmath>

extern "C" {
    PyObject* PyInit__C(void) {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_C",
            "This module is a encoder kernel register trigger.\n\nYou should never use this module. Before import this, you should import torch first.",
            -1,
            NULL,
        };
        return PyModule_Create(&module_def);
    }
}

namespace encoder {


    std::tuple<torch::Tensor, torch::Tensor> encoder_cpu_forward(
        torch::Tensor wheel_speeds,
        torch::Tensor remaining_clicks,
        torch::Tensor rw_signal_state,
        torch::Tensor converted,
        double clicks_per_radian,
        double dt) {
        
        // Input validation
        TORCH_CHECK(wheel_speeds.is_contiguous(), "wheel_speeds must be contiguous");
        TORCH_CHECK(remaining_clicks.is_contiguous(), "remaining_clicks must be contiguous");
        TORCH_CHECK(rw_signal_state.is_contiguous(), "rw_signal_state must be contiguous");
        TORCH_CHECK(converted.is_contiguous(), "converted must be contiguous");

        // Ensure 1D tensors
        wheel_speeds = wheel_speeds.flatten();
        remaining_clicks = remaining_clicks.flatten();
        rw_signal_state = rw_signal_state.flatten();
        converted = converted.flatten();

        // Get tensor dimensions
        const int size = wheel_speeds.numel();
        
        // Check all tensors have same size
        TORCH_CHECK(remaining_clicks.numel() == size, "remaining_clicks size mismatch");
        TORCH_CHECK(rw_signal_state.numel() == size, "rw_signal_state size mismatch");
        TORCH_CHECK(converted.numel() == size, "converted size mismatch");
        
        // Create output tensors
        auto options = torch::TensorOptions().dtype(wheel_speeds.dtype()).device(torch::kCPU);
        torch::Tensor new_output = torch::zeros({size}, options);
        torch::Tensor new_remaining_clicks = torch::zeros({size}, options);

        // Access tensor data
        auto wheel_speeds_acc = wheel_speeds.accessor<float, 1>();
        auto remaining_clicks_acc = remaining_clicks.accessor<float, 1>();
        auto rw_signal_state_acc = rw_signal_state.accessor<float, 1>();
        auto converted_acc = converted.accessor<float, 1>();
        auto new_output_acc = new_output.accessor<float, 1>();
        auto new_remaining_clicks_acc = new_remaining_clicks.accessor<float, 1>();

        // CPU processing loop
        for (int i = 0; i < size; ++i) {
            float signal_state = rw_signal_state_acc[i];

            if (signal_state == SIGNAL_NOMINAL) {
                // SIGNAL_NOMINAL processing
                float angle = wheel_speeds_acc[i] * dt;
                float temp = angle * clicks_per_radian + remaining_clicks_acc[i];
                float number_clicks = std::trunc(temp);
                new_remaining_clicks_acc[i] = temp - number_clicks;
                new_output_acc[i] = number_clicks / (clicks_per_radian * dt);
            }
            else if (signal_state == SIGNAL_OFF) {
                // SIGNAL_OFF processing
                new_remaining_clicks_acc[i] = 0.0f;
                new_output_acc[i] = 0.0f;
            }
            else if (signal_state == SIGNAL_STUCK) {
                // SIGNAL_STUCK processing
                new_remaining_clicks_acc[i] = remaining_clicks_acc[i];
                new_output_acc[i] = converted_acc[i];
            }
            else {
                // Default case
                new_remaining_clicks_acc[i] = remaining_clicks_acc[i];
                new_output_acc[i] = wheel_speeds_acc[i];
            }
        }

        return {new_output, new_remaining_clicks};
    }
    TORCH_LIBRARY(encoder, m) {
        m.def("encoder_kernel(Tensor wheel_speeds, Tensor remaining_clicks, Tensor rw_signal_state, Tensor converted, float clicks_per_radian, float dt) -> (Tensor, Tensor)");
    }
    
    TORCH_LIBRARY_IMPL(encoder, CPU, m) {
        m.impl("encoder_kernel", &encoder_cpu_forward);
    }
}

