'''This Module perform register call for python and c++/cuda implementation of encoder kernel.

In no circumstance should you use anything from this module. 
All function should be called from torch.ops.encoder namespace.

All available operator:
torch.ops.encoder.c
torch.ops.encoder.py_
'''

__all__ = []

import torch

from . import _C
from .encoder import WheelSpeedEncoderSignal


def _backward(
    ctx: torch.autograd.function.BackwardCFunction,
    grad_new_output: torch.Tensor,
    grad_new_remaining_clicks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, None, None, None, None]:
    _, _, signals, _ = ctx.saved_tensors

    grad_target_speeds = None
    grad_remaining_clicks = None
    grad_speeds = None

    if ctx.needs_input_grad[0]:
        grad_target_speeds = torch.where(
            signals == WheelSpeedEncoderSignal.NOMINAL,
            grad_new_output,
            0.,
        )

    if ctx.needs_input_grad[1]:
        grad_remaining_clicks = torch.where(
            signals == WheelSpeedEncoderSignal.NOMINAL,
            grad_new_remaining_clicks,
            0.,
        )

    if ctx.needs_input_grad[3]:
        grad_speeds = torch.where(
            signals == WheelSpeedEncoderSignal.LOCKED,
            grad_new_output,
            0.,
        )
    return grad_target_speeds, grad_remaining_clicks, None, grad_speeds, None, None


def _setup_context(
    ctx: torch.autograd.function.BackwardCFunction,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                  float, float],
    output: tuple[torch.Tensor, torch.Tensor],
) -> None:
    target_speeds, remaining_clicks, signals, speeds, clicks_per_radian, dt = inputs
    ctx.save_for_backward(target_speeds, remaining_clicks, signals, speeds)
    ctx.clicks_per_radian = clicks_per_radian
    ctx.dt = dt


@torch.library.custom_op("encoder::py_", mutates_args=[])
def encoder_py(
    target_speeds: torch.Tensor,
    remaining_clicks: torch.Tensor,
    signals: torch.Tensor,
    speeds: torch.Tensor,
    clicks_per_radian: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    speeds = speeds.clone()

    mask = signals == WheelSpeedEncoderSignal.NOMINAL
    if torch.any(mask):
        target_radian = target_speeds * dt
        target_clicks = (target_radian * clicks_per_radian + remaining_clicks)

        remaining_clicks = torch.where(
            mask,
            target_clicks % 1,
            remaining_clicks,
        )

        speeds = torch.where(
            mask,
            torch.floor(target_clicks) / (dt * clicks_per_radian),
            speeds,
        )

    mask = signals == WheelSpeedEncoderSignal.STOPPED
    if mask.any():
        remaining_clicks = torch.where(
            mask,
            0.,
            remaining_clicks,
        )
        speeds = torch.where(
            mask,
            0.,
            speeds,
        )

    if torch.all(signals == WheelSpeedEncoderSignal.LOCKED):
        # if all signal is locked, clone to avoid mutates args error
        remaining_clicks = remaining_clicks.clone()

    return speeds, remaining_clicks


torch.library.register_autograd("encoder::c",
                                _backward,
                                setup_context=_setup_context)

torch.library.register_autograd("encoder::py_",
                                _backward,
                                setup_context=_setup_context)
