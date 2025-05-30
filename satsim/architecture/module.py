from typing import Any
from abc import ABC, abstractmethod

from torch import nn

from .timer import Timer


class Module(nn.Module, ABC):

    def __init__(self, *args, timer: Timer, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._timer = timer

    @abstractmethod
    def _forward(self, *args, **kwargs) -> Any:
        pass

    def forward(self, *args, **kwargs) -> Any:
        return self._forward(*args, **kwargs)

    def reset(self) -> None:
        module: Module
        for module in self.children():
            module.reset()
