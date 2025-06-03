__all__ = ['Module']
from abc import ABC, abstractmethod

from typing import Any, Generic, TypeVar, cast
from torch import nn
import torch

from .timer import Timer

T = TypeVar('T', bound=dict[str, Any])


class Module(nn.Module, ABC, Generic[T]):

    def __init__(self, *args, timer: Timer, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._timer = timer

    @abstractmethod
    def forward(self, state_dict: T, *args,
                **kwargs) -> tuple[T, tuple[Any, ...]]:
        pass

    def reset(self) -> T:
        state_dict: dict[str, Any] = {}
        for name, module in self.named_children():
            module = cast(Module, module)
            state_dict[name] = module.reset()

        return state_dict
