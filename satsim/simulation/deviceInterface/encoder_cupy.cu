const int SIGNAL_NOMINAL    = 0;
const int SIGNAL_OFF        = 1;
const int SIGNAL_STUCK      = 2;
extern "C" __global__
void cupy_encoder_kernel(const float* wheel_speeds, const float* remaining_clicks,
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
