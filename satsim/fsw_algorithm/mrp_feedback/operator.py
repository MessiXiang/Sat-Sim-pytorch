__all__ = ['softclamp']
import torch


class Softclamp(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.BackwardCFunction,
        input: torch.Tensor,
        max: float,
        min: float,
    ):
        ctx.save_for_backward(input)
        ctx.max_ = max
        ctx.min_ = min
        return torch.clamp(input, max=max, min=min)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.BackwardCFunction,
        grad_output: torch.Tensor,
    ):
        input_, = ctx.saved_tensors
        max_, min_ = ctx.max_, ctx.min_
        grad_input = grad_output * torch.where(
            (input_ > max_) | (input_ < min_),
            0.1,
            1.,
        )
        return grad_input, None, None


def softclamp(input: torch.Tensor, max: torch.Tensor, min: torch.Tensor):
    return Softclamp.apply(input, max, min)
