__all__ = ['move_to', 'dict_recursive_apply', 'make_dict_copy']
from functools import partial
from typing import Any, Callable

import torch


def move_to(
    state_dict: dict[str, torch.Tensor | dict],
    target: str | torch.dtype | torch.Tensor,
) -> dict[str, torch.Tensor | dict]:
    _move = lambda x: x.to(target)
    return dict_recursive_apply(state_dict=state_dict, fn=_move)


def dict_recursive_apply(
    state_dict: dict[str, Any],
    fn: Callable[[Any], Any],
) -> dict[str, Any]:

    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            state_dict[key] = fn(value)
        else:
            state_dict[key] = dict_recursive_apply(value, fn)
    return state_dict


def make_dict_copy(d: dict[str, Any], ) -> dict[str, Any]:
    new_dict = dict()
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.clone()
        else:
            new_dict[k] = make_dict_copy(v)

    return new_dict
