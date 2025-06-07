import torch
from . import _C
from .encoder import *


def _backward(ctx: torch.autograd.function.BackwardCFunction,
              grad_new_output: torch.Tensor,
              grad_new_remaining_clicks: torch.Tensor):
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


torch.library.register_autograd("encoder::encoder_kernel",
                                _backward,
                                setup_context=_setup_context)
