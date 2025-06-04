__all__ = [
    'Encoder',
    'encoder_signal',
    'EncoderState',
]
from json import encoder
from typing import Any, TypedDict, cast
from enum import IntEnum
import torch
import cupy
import pathlib

from satsim.architecture import (
    Module,
    Timer,
)


class encoder_signal(IntEnum):
    NOMINAL = 0
    OFF = 1
    STUCK = 2


class EncoderState(TypedDict):
    """encoder state dict"""
    rw_signal_state: torch.Tensor
    remaining_clicks: torch.Tensor
    converted: torch.Tensor


# Define unified CuPy kernel for all encoder signal states
cupy_encoder_kernel = cupy.RawKernel(
    pathlib.Path(__file__).with_suffix('.cu').read_text(),
    'cupy_encoder_kernel',
)


class UnifiedEncoderFunction(torch.autograd.Function):
    """Unified autograd function for all encoder signal states."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.BackwardCFunction,
        wheel_speeds: torch.Tensor,
        remaining_clicks: torch.Tensor,
        rw_signal_state: torch.Tensor,
        converted: torch.Tensor,
        clicks_per_radian: float,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(wheel_speeds, remaining_clicks, rw_signal_state,
                              converted)
        ctx.clicks_per_radian = clicks_per_radian
        ctx.dt = dt

        cupy_wheel_speeds = cupy.ascontiguousarray(
            cupy.from_dlpack(wheel_speeds.detach().cuda()))
        cupy_remaining_clicks = cupy.ascontiguousarray(
            cupy.from_dlpack(remaining_clicks.detach().cuda()))
        cupy_rw_signal_state = cupy.ascontiguousarray(
            cupy.from_dlpack(rw_signal_state.detach().cuda()))
        cupy_converted = cupy.ascontiguousarray(
            cupy.from_dlpack(converted.detach().cuda()))

        cupy_new_output = cupy.empty_like(cupy_wheel_speeds)
        cupy_new_remaining_clicks = cupy.empty_like(cupy_remaining_clicks)

        size = cupy_wheel_speeds.size
        block_sum = 128

        print(clicks_per_radian)
        cupy_encoder_kernel(
            (block_sum, ), ((size + block_sum - 1) // block_sum, ),
            (cupy_wheel_speeds, cupy_remaining_clicks, cupy_rw_signal_state,
             cupy_converted, cupy_new_output, cupy_new_remaining_clicks,
             cupy.float32(clicks_per_radian), cupy.float32(dt),
             cupy.int32(size)))

        torch_new_output = torch.from_dlpack(cupy_new_output).cpu()
        torch_new_remaining_clicks = torch.from_dlpack(
            cupy_new_remaining_clicks).cpu()

        return torch_new_output, torch_new_remaining_clicks

    @staticmethod
    def backward(
        ctx: torch.autograd.function.BackwardCFunction,
        grad_new_output: torch.Tensor,
        grad_new_remaining_clicks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None, None]:
        _, _, rw_signal_state, _ = ctx.saved_tensors

        grad_wheel_speeds = torch.where(
            rw_signal_state == encoder_signal.NOMINAL, grad_new_output,
            torch.zeros_like(grad_new_output))

        grad_remaining_clicks = torch.where(
            rw_signal_state == encoder_signal.NOMINAL,
            grad_new_remaining_clicks,
            torch.zeros_like(grad_new_remaining_clicks))

        return grad_wheel_speeds, grad_remaining_clicks, None, None, None, None


class Encoder(Module):
    """encoder module"""

    def __init__(
        self,
        *args,
        timer: Timer,
        numRW: int,
        clicksPerRotation: int,
        **kwargs,
    ) -> None:

        super().__init__(*args, timer=timer, **kwargs)

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

    def forward(
        self,
        state_dict: EncoderState,
        *args,
        wheel_speeds: torch.Tensor,
        **kwargs,
    ) -> tuple[EncoderState, tuple[torch.Tensor]]:
        if self._timer.step_count == 0:
            state_dict['converted'] = wheel_speeds
            return state_dict, (wheel_speeds, )

        assert torch.isin(
            state_dict['rw_signal_state'],
            torch.tensor([
                encoder_signal.NOMINAL, encoder_signal.OFF,
                encoder_signal.STUCK
            ])).all(), "encoder: un-modeled encoder signal mode selected."

        new_output, new_remaining_clicks = UnifiedEncoderFunction.apply(  # type:ignore
            wheel_speeds, state_dict['remaining_clicks'],
            state_dict['rw_signal_state'], state_dict['converted'],
            self._clicks_per_radian, self._timer.dt)

        state_dict['remaining_clicks'] = new_remaining_clicks
        state_dict['converted'] = new_output

        return state_dict, (new_output, )
