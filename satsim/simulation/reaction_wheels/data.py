__all__ = [
    'ReactionWheelModels',
    'ReactionWheelsOutput',
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


class ReactionWheelsOutput(TypedDict):
    rWB_B: torch.Tensor  # [3]
    gsHat_B: torch.Tensor  # [3]
    w2Hat0_B: torch.Tensor  # [3]
    w3Hat0_B: torch.Tensor  # [3]
    mass: torch.Tensor
    theta: torch.Tensor
    Omega: torch.Tensor
    Js: torch.Tensor
    Jt: torch.Tensor
    Jg: torch.Tensor
    U_s: torch.Tensor
    U_d: torch.Tensor
    d: torch.Tensor
    J13: torch.Tensor
    u_current: torch.Tensor
    frictionTorque: torch.Tensor
    u_max: torch.Tensor
    u_min: torch.Tensor
    u_f: torch.Tensor
    Omega_max: torch.Tensor
    P_max: torch.Tensor
    linearFrictionRatio: torch.Tensor
    RWModel: torch.Tensor


class ReactionWheelDynamicParams(TypedDict):
    omega: torch.Tensor
    delta_omega: torch.Tensor
