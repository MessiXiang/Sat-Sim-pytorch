__all__ = [
    'Module',
]

from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, TypeVar, cast

from torch import nn

from .timer import Timer

T = TypeVar('T', bound=Mapping[str, Any])


class Module(nn.Module, ABC, Generic[T]):

    def __init__(self, *args, timer: Timer, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._timer = timer

    @abstractmethod
    def forward(
        self,
        state_dict: T,
        *args,
        **kwargs,
    ) -> tuple[T, tuple[Any, ...]]:
        pass

    def reset(self) -> T:
        state_dict: dict[str, Any] = dict()
        for name, child in self.named_children():
            child = cast(Module, child)
            state_dict[name] = child.reset()
        return cast(T, state_dict)
