__all__ = [
    'ReactionWheelModels',
    'ReactionWheelDynamicParams',
]
from enum import IntEnum, auto
from typing import TypedDict

import torch


class ReactionWheelModels(IntEnum):
    BALANCED_WHEELS = auto()
    JITTER_SIMPLE = auto()
    JITTER_FULLY_COUPLED = auto()

    @classmethod
    def is_jitter(cls, model: int) -> bool:
        return model == cls.JITTER_FULLY_COUPLED or model == cls.JITTER_SIMPLE


class ReactionWheelDynamicParams(TypedDict):
    omega: torch.Tensor
    delta_omega: torch.Tensor
