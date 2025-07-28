__all__ = ['move_to', 'recursive_apply']
from functools import partial
from typing import Callable
import torch


def move_to(
    state_dict: dict[str, torch.Tensor | dict],
    target: str | torch.dtype | torch.Tensor,
) -> dict[str, torch.Tensor | dict]:
    _move = lambda x: x.to(target)
    return recursive_apply(state_dict=state_dict, fn=_move)


def recursive_apply(
    state_dict: dict[str, torch.Tensor | dict],
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, torch.Tensor | dict]:

    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            state_dict[key] = fn(value)
        else:
            state_dict[key] = recursive_apply(value, fn)
    return state_dict
