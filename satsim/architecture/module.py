__all__ = ['Module', 'VoidStateDict', 'ModuleList']

from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, TypeVar, TypedDict, cast

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


# NOTE: Module should be placed before nn.ModuleList, because nn.ModuleList
#  doesn't support extending __init__
class ModuleList(Module[T], nn.ModuleList):

    def forward(
        self,
        state_dict: T,
        *args,
        **kwargs,
    ) -> tuple[T, tuple[Any, ...]]:
        raise NotImplementedError("ModuleList should never been called")


class VoidStateDict(TypedDict):
    pass
