__all__ = ['Encoder']
from tkinter import NO
from typing import Any, TypedDict, cast
import torch
import cupy

from ...architecture import (
    SIGNAL_NOMINAL,
    SIGNAL_OFF,
    SIGNAL_STUCK,
    Module,
    Timer,
)

# Define CuPy kernel for SIGNAL_NOMINAL forward computation
cupy_nominal_kernel_fwd = cupy.RawKernel(
    r"""
extern "C" __global__
void cupy_nominal_kernel_fwd(const float* wheel_speeds, const float* remaining_clicks,
                            float* new_output, float* new_remaining_clicks,
                            float clicks_per_radian, float dt, int size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size) {
        float angle = wheel_speeds[tid] * dt;
        float temp = angle * clicks_per_radian + remaining_clicks[tid];
        float number_clicks = trunc(temp);
        new_remaining_clicks[tid] = temp - number_clicks;
        new_output[tid] = number_clicks / (clicks_per_radian * dt);
    }
}
""",
    'cupy_nominal_kernel_fwd',
)


class NominalEncoderFunction(torch.autograd.Function):
    """class to do autograd"""

    @staticmethod
    def forward(
        ctx,
        wheel_speeds: torch.Tensor,
        remaining_clicks: torch.Tensor,
        clicks_per_radian: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(wheel_speeds, remaining_clicks,
                              clicks_per_radian, dt)
        cupy_wheel_speeds = cupy.ascontiguousarray(
            cupy.from_dlpack(wheel_speeds.detach()))
        cupy_remaining_clicks = cupy.ascontiguousarray(
            cupy.from_dlpack(remaining_clicks.detach()))
        cupy_new_output = cupy.empty_like(cupy_wheel_speeds)
        cupy_new_remaining_clicks = cupy.empty_like(cupy_remaining_clicks)

        size = cupy_wheel_speeds.size
        bs = 128
        cupy_nominal_kernel_fwd(
            (bs, ), ((size + bs - 1) // bs, ),
            (cupy_wheel_speeds, cupy_remaining_clicks, cupy_new_output,
             cupy_new_remaining_clicks, clicks_per_radian, dt, size))

        torch_new_output = torch.from_dlpack(cupy_new_output)
        torch_new_remaining_clicks = torch.from_dlpack(
            cupy_new_remaining_clicks)
        return torch_new_output, torch_new_remaining_clicks

    @staticmethod
    def backward(
        ctx,
        grad_new_output: torch.Tensor,
        _: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:
        _, _, _, _ = ctx.saved_tensors
        grad_wheel_speeds = grad_new_output
        return grad_wheel_speeds, None, None, None


class EncoderState(TypedDict):
    """encoder state dict"""
    rw_signal_state: torch.Tensor
    remaining_clicks: torch.Tensor
    converted: torch.Tensor


class Encoder(Module):
    """encoder module"""

    def __init__(self,
                 *args,
                 timer: Timer,
                 numRW: int = -1,
                 clicksPerRotation: int = -1,
                 **kwargs) -> None:

        super().__init__(*args, timer=timer, **kwargs)

        assert clicksPerRotation > 0, "encoder: number of clicks must be a positive integer."
        assert numRW > 0, "encoder: number of reaction wheels must be a \
            positive integer. It may not have been set."

        self._num_rw = numRW
        self._clicks_per_rotation = clicksPerRotation

    @property
    def _clicks_per_radian(self) -> float:
        return self._clicks_per_rotation / (2 * torch.pi)

    def reset(self) -> EncoderState:
        state_dict: dict[str, Any] = super().reset()
        state_dict['rw_signal_state'] = torch.zeros(self._num_rw)
        state_dict['remaining_clicks'] = torch.zeros(self._num_rw)
        state_dict['converted'] = torch.zeros(self._num_rw)
        return cast(EncoderState, state_dict)

    def forward(self, state_dict: EncoderState, *args,
                wheel_speeds: torch.Tensor, **kwargs) -> torch.Tensor:
        if self._timer.step_count == 0:
            return wheel_speeds

        new_output = torch.zeros_like(wheel_speeds)

        assert torch.isin(
            state_dict['rw_signal_state'],
            torch.tensor([
                SIGNAL_NOMINAL, SIGNAL_OFF, SIGNAL_STUCK
            ])).all(), "encoder: un-modeled encoder signal mode selected."

        signal_nominal_mask = state_dict['rw_signal_state'] == SIGNAL_NOMINAL
        if torch.any(signal_nominal_mask):
            nominal_wheel_speeds = wheel_speeds[signal_nominal_mask]
            nominal_remaining_clicks = state_dict['remaining_clicks'][
                signal_nominal_mask]
            nominal_new_output, nominal_new_remaining_clicks = NominalEncoderFunction.apply(
                nominal_wheel_speeds, nominal_remaining_clicks,
                self._clicks_per_radian, self._timer.dt)
            new_output[signal_nominal_mask] = nominal_new_output
            state_dict['remaining_clicks'][
                signal_nominal_mask] = nominal_new_remaining_clicks

        signal_off_mask = state_dict['rw_signal_state'] == SIGNAL_OFF
        if torch.any(signal_off_mask):
            state_dict['remaining_clicks'][signal_off_mask] = 0.0
            new_output[signal_off_mask] = 0.0

        signal_stuck_mask = state_dict['rw_signal_state'] == SIGNAL_STUCK
        if torch.any(signal_stuck_mask):
            new_output[signal_stuck_mask] = state_dict['converted'][
                signal_stuck_mask]

        state_dict['converted'] = new_output
        return new_output
