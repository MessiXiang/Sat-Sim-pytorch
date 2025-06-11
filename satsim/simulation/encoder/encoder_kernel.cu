#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "encoder.h"

namespace encoder {
    __global__ void kernel(
        const float* target_speeds, 
        const float* remaining_clicks,
        const int* signals, const float* speeds,
        float* new_speeds, float* new_remaining_clicks,
        float clicks_per_radian, float dt, int size
    ) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= size) {
            return;
        }

        switch (signals[tid]) {
        case int(ReactionWheelSignal::NOMINAL):
            float target_radian = target_speeds[tid] * dt;
            float target_clicks = target_radian * clicks_per_radian + remaining_clicks[tid];
            float number_clicks = floor(target_clicks);
            new_remaining_clicks[tid] = target_clicks - number_clicks;
            new_speeds[tid] = number_clicks / (clicks_per_radian * dt);
            return;
        case int(ReactionWheelSignal::STOPPED):
            new_remaining_clicks[tid] = 0.0f;
            new_speeds[tid] = 0.0f;
            return;
        case int(ReactionWheelSignal::LOCKED):
            new_remaining_clicks[tid] = remaining_clicks[tid]; 
            new_speeds[tid] = speeds[tid];
            return;
        default:
            assert(false);
        }
    }

    // #define CUDA_CHECK(call) \
    //     do { \
    //         cudaError_t error = call; \
    //         if (error != cudaSuccess) { \
    //             fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
    //                     cudaGetErrorString(error)); \
    //             exit(1); \
    //         } \
    //     } while(0)

    // C++ forward_cuda function
    std::tuple<torch::Tensor, torch::Tensor> forward_cuda(
        torch::Tensor target_speeds,
        torch::Tensor remaining_clicks,
        torch::Tensor signals,
        torch::Tensor speeds,
        double clicks_per_radian,
        double dt) {
        
        TORCH_CHECK(target_speeds.is_cuda(), "target_speeds must be a CUDA tensor");
        TORCH_CHECK(remaining_clicks.is_cuda(), "remaining_clicks must be a CUDA tensor");
        TORCH_CHECK(signals.is_cuda(), "signals must be a CUDA tensor");
        TORCH_CHECK(speeds.is_cuda(), "speeds must be a CUDA tensor");
        
        target_speeds = target_speeds.contiguous();
        remaining_clicks = remaining_clicks.contiguous();
        signals = signals.contiguous();
        speeds = speeds.contiguous();
        
        torch::Tensor new_speeds = torch::zeros_like(speeds);
        torch::Tensor new_remaining_clicks = torch::zeros_like(remaining_clicks);
        
        const int size = target_speeds.numel();

        const int threads_per_block = 32;
        const int blocks = (size + threads_per_block - 1) / threads_per_block;
        kernel<<<blocks, threads_per_block>>>(
            target_speeds.data_ptr<float>(),
            remaining_clicks.data_ptr<float>(),
            signals.data_ptr<int>(),
            speeds.data_ptr<float>(),
            new_speeds.data_ptr<float>(),
            new_remaining_clicks.data_ptr<float>(),
            clicks_per_radian,
            dt,
            size
        );
        
        TORCH_CHECK(!cudaGetLastError());
        TORCH_CHECK(!cudaDeviceSynchronize());
        
        return {new_speeds, new_remaining_clicks};
    }
    

    TORCH_LIBRARY_IMPL(encoder, CUDA, m) {
        m.impl("c", &forward_cuda);
    }
}
