'''This is register call for python and c++/cuda implementation of encoder kernel.

In no circumstance should you use anything from this module. 
All function should be called from torch.ops.encoder namespace.

All available operator:
torch.ops.encoder.encoder_kernel
torch.ops.encoder.encoder_py
'''
__all__ = []
import torch

from . import _C
from .encoder import EncoderSignal


def _backward(
    ctx: torch.autograd.function.BackwardCFunction,
    grad_new_output: torch.Tensor,
    grad_new_remaining_clicks: torch.Tensor,
):
    _, _, rw_signal_state, _ = ctx.saved_tensors

    grad_wheel_speeds = torch.where(
        rw_signal_state == EncoderSignal.NOMINAL, grad_new_output,
        torch.zeros_like(grad_new_output)) if ctx.needs_input_grad[0] else None

    grad_remaining_clicks = torch.where(
        rw_signal_state == EncoderSignal.NOMINAL, grad_new_remaining_clicks,
        torch.zeros_like(
            grad_new_remaining_clicks)) if ctx.needs_input_grad[1] else None

    return grad_wheel_speeds, grad_remaining_clicks, None, None, None, None


def _setup_context(
    ctx: torch.autograd.function.BackwardCFunction,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                  float, float],
    output: tuple[torch.Tensor, torch.Tensor],
):
    wheel_speeds, remaining_clicks, rw_signal_state, converted, clicks_per_radian, dt = inputs
    ctx.save_for_backward(wheel_speeds, remaining_clicks, rw_signal_state,
                          converted)
    ctx.clicks_per_radian = clicks_per_radian
    ctx.dt = dt


@torch.library.custom_op("encoder::encoder_py", mutates_args=())
def encoder_py(
    wheel_speeds: torch.Tensor,
    remaining_clicks: torch.Tensor,
    rw_signal_state: torch.Tensor,
    converted: torch.Tensor,
    clicks_per_radian: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:

    new_output = torch.zeros_like(wheel_speeds)

    signal_nominal_mask = rw_signal_state == EncoderSignal.NOMINAL
    if torch.any(signal_nominal_mask):
        angle = wheel_speeds * dt
        number_clicks = torch.trunc(angle * clicks_per_radian +
                                    remaining_clicks)

        remaining_clicks = torch.where(
            signal_nominal_mask,
            angle * clicks_per_radian + remaining_clicks[signal_nominal_mask] -
            number_clicks,
            remaining_clicks,
        )

        new_output = torch.where(signal_nominal_mask,
                                 number_clicks / (clicks_per_radian * dt),
                                 new_output)

    ## SIGNAL_OFF_SITUATION

    signal_off_mask = rw_signal_state == EncoderSignal.OFF
    if torch.any(signal_off_mask):
        remaining_clicks = torch.where(signal_off_mask, 0., remaining_clicks)
        new_output = torch.where(signal_off_mask, 0., new_output)

    ## SIGNAL_STUCK_SITUATION
    signal_stuck_mask = rw_signal_state == EncoderSignal.STUCK
    if torch.any(signal_stuck_mask):
        new_output = torch.where(signal_stuck_mask, converted, new_output)
        remaining_clicks = remaining_clicks.clone()
    return new_output, remaining_clicks


torch.library.register_autograd("encoder::encoder_kernel",
                                _backward,
                                setup_context=_setup_context)

torch.library.register_autograd("encoder::encoder_py",
                                _backward,
                                setup_context=_setup_context)
