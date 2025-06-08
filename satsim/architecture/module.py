__all__ = ['Module']
from abc import ABC, abstractmethod

from typing import Any, Generic, Mapping, TypeVar, cast
from torch import nn
import torch

from .timer import Timer

T = TypeVar('T', bound=Mapping[str, Any])


class Module(nn.Module, ABC, Generic[T]):

    def __init__(self, *args, timer: Timer, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._timer = timer

    @abstractmethod
    def forward(self, state_dict: T, *args,
                **kwargs) -> tuple[T, tuple[Any, ...]]:
        pass

    def reset(self, device: str = 'cpu') -> Mapping[str, Any]:
        state_dict: dict[str, Any] = {}
        for name, module in self.named_children():
            module = cast(Module, module)
            state_dict[name] = module.reset()

        return self.move_state_to(state_dict, device)

    def move_state_to(self, state_dict: Mapping[str, Any],
                      device: str) -> Mapping[str, Any]:
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                state_dict[key] = value.to(device)
            elif isinstance(value, Mapping):
                state_dict[key] = self.move_state_to(value, device)

        return state_dict
