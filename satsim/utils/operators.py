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
    """Execute a custom PyTorch operator and return the result.

    This function leverages Python's `inspect` module to dynamically resolve
    and invoke custom operators registered via `torch.library`. 
    Users must ensure the following consistency:
    1. The namespace specified during `torch.library` registration matches 
       the filename of the module containing the operator definition
    2. The operator name exactly matches the symbol registered with torch.library
    3. The namespace has two operator respectively named as 'py_' and 'c'

    Parameters:
        *args: Positional arguments to be forwarded to the target operator
        **kwargs: Keyword arguments to be forwarded to the target operator

    Returns:
        Any: The output(s) produced by the target operator

    Notes:
        - Operators must be registered via `torch.library` prior to invocation
        - Module filenames must strictly match the registered namespace (case-sensitive)
        - This function relies on call stack inspection and should be invoked 
          from the top-level module environment
        - Which implementation to use is determined by environ 'OPTIMIZE'
        - This function won't work if it's called from __main__ 
    
    Examples:
        1. Define and register a custom operator in `operator.py`:
            >>> import torch
            >>> @torch.library.custom_op("sin::py_", mutates_args=[])
            >>> def sin(x: torch.Tensor) -> torch.Tensor:
            >>>     return x.sin()

        2. Use this function from a module named as 'sin.py':
            >>> import torch
            >>> from satsim.utils import run_operator
            >>> run_operator(torch.zeros(3))
            tensor([0., 0., 0.])

    Raises:
        RuntimeError: If operator resolution fails due to naming inconsistencies
    """
    stack = inspect.stack()
    frame, *_ = stack[1]
    module = inspect.getmodule(frame)
    assert module is not None
    *_, module_name = module.__name__.split('.')
    operator_module = getattr(torch.ops, module_name)
    operator = operator_module.c if Store.OPTIMIZE else operator_module.py_
    return operator(*args, **kwargs)
