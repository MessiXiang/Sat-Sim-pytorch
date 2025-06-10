__all__ = [
    'run_operator',
]

import inspect
from typing import Any

import todd
import torch


class Store(todd.Store):
    OPTIMIZE: bool


def run_operator(*args, **kwargs) -> Any:
    stack = inspect.stack()
    frame, *_ = stack[1]
    module = inspect.getmodule(frame)
    assert module is not None
    *_, module_name = module.__name__.split('.')
    operator_module = getattr(torch.ops, module_name)
    operator = operator_module.c if Store.OPTIMIZE else operator_module.py_
    return operator(*args, **kwargs)
