#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// extern "C" {
//     /* Creates a dummy empty _C module that can be imported from Python.
//     The import from Python will load the .so consisting of this file
//     in this extension, so that the TORCH_LIBRARY static initializers
//     below are run. */
//     PyObject* PyInit__C(void)
//     {
//         static struct PyModuleDef module_def = {
//             PyModuleDef_HEAD_INIT,
//             "_C",   /* name of module */
//             NULL,   /* module documentation, may be NULL */
//             -1,     /* size of per-interpreter state of the module,
//                         or -1 if the module keeps state in global variables. */
//             NULL,   /* methods */
//         };
//         return PyModule_Create(&module_def);
//     }
// }


// Signal state constants
const int SIGNAL_NOMINAL = 0;
const int SIGNAL_OFF     = 1;
const int SIGNAL_STUCK   = 2;

// CUDA kernel
namespace encoder{
    __global__ void cuda_encoder_kernel(
        const float* wheel_speeds, const float* remaining_clicks,
        const float* rw_signal_state, const float* converted,
        float* new_output, float* new_remaining_clicks,
        float clicks_per_radian, float dt, int size) {
        
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < size) {
            float signal_state = rw_signal_state[tid];

            if (signal_state == SIGNAL_NOMINAL) {
                // SIGNAL_NOMINAL processing
                float angle = wheel_speeds[tid] * dt;
                float temp = angle * clicks_per_radian + remaining_clicks[tid];
                float number_clicks = trunc(temp);
                new_remaining_clicks[tid] = temp - number_clicks;
                new_output[tid] = number_clicks / (clicks_per_radian * dt);
            }
            else if (signal_state == SIGNAL_OFF) {
                // SIGNAL_OFF processing
                new_remaining_clicks[tid] = 0.0f;
                new_output[tid] = 0.0f;
            }
            else if (signal_state == SIGNAL_STUCK) {
                // SIGNAL_STUCK processing
                new_remaining_clicks[tid] = remaining_clicks[tid]; // keep unchanged
                new_output[tid] = converted[tid];
            }
            else {
                // Default case (should not happen if input is valid)
                new_remaining_clicks[tid] = remaining_clicks[tid];
                new_output[tid] = wheel_speeds[tid];
            }
        }
    }

    // CUDA error checking macro
    #define CUDA_CHECK(call) \
        do { \
            cudaError_t error = call; \
            if (error != cudaSuccess) { \
                fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                        cudaGetErrorString(error)); \
                exit(1); \
            } \
        } while(0)

    // C++ wrapper function
    std::tuple<torch::Tensor, torch::Tensor> encoder_cuda_forward(
        torch::Tensor wheel_speeds,
        torch::Tensor remaining_clicks,
        torch::Tensor rw_signal_state,
        torch::Tensor converted,
        double clicks_per_radian,
        double dt) {
        
        // Check inputs are on CUDA
        TORCH_CHECK(wheel_speeds.is_cuda(), "wheel_speeds must be a CUDA tensor");
        TORCH_CHECK(remaining_clicks.is_cuda(), "remaining_clicks must be a CUDA tensor");
        TORCH_CHECK(rw_signal_state.is_cuda(), "rw_signal_state must be a CUDA tensor");
        TORCH_CHECK(converted.is_cuda(), "converted must be a CUDA tensor");
        
        // Check inputs are contiguous
        wheel_speeds = wheel_speeds.contiguous();
        remaining_clicks = remaining_clicks.contiguous();
        rw_signal_state = rw_signal_state.contiguous();
        converted = converted.contiguous();
        
        // Get tensor dimensions
        const int size = wheel_speeds.numel();
        
        // Check all tensors have same size
        TORCH_CHECK(remaining_clicks.numel() == size, "remaining_clicks size mismatch");
        TORCH_CHECK(rw_signal_state.numel() == size, "rw_signal_state size mismatch");
        TORCH_CHECK(converted.numel() == size, "converted size mismatch");
        
        // Create output tensors
        auto options = torch::TensorOptions().dtype(wheel_speeds.dtype()).device(wheel_speeds.device());
        torch::Tensor new_output = torch::zeros({size}, options);
        torch::Tensor new_remaining_clicks = torch::zeros({size}, options);
        
        // Launch configuration
        const int threads_per_block = 256;
        const int blocks = (size + threads_per_block - 1) / threads_per_block;
        
        // Launch kernel
        cuda_encoder_kernel<<<blocks, threads_per_block>>>(
            wheel_speeds.data_ptr<float>(),
            remaining_clicks.data_ptr<float>(),
            rw_signal_state.data_ptr<float>(),
            converted.data_ptr<float>(),
            new_output.data_ptr<float>(),
            new_remaining_clicks.data_ptr<float>(),
            clicks_per_radian,
            dt,
            size
        );
        
        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        return {new_output, new_remaining_clicks};
    }
    

    TORCH_LIBRARY_IMPL(encoder, CUDA, m) {
        m.impl("encoder_kernel", &encoder_cuda_forward);
    }
}
