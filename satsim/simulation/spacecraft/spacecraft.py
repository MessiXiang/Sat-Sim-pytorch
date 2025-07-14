from typing import TypedDict

import torch

from satsim.architecture import Module
from .hub_effector import HubEffector

from ..base import BackSubMatrices


class SpacecraftStateDict(TypedDict):
    dvAccum_CN_B: torch.Tensor
    dvAccum_BN_B: torch.Tensor
    dvAccum_CN_N: torch.Tensor


class Spacecraft(Module[SpacecraftStateDict]):

    def __init__(
        self,
        *args,
        hub: HubEffector,
        dvAccum_CN_B: torch.Tensor | None = None,
        dvAccum_BN_B: torch.Tensor | None = None,
        dvAccum_CN_N: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dvAccum_CN_B = dvAccum_CN_B or torch.zeros(3)
        self.dvAccum_BN_B = dvAccum_BN_B or torch.zeros(3)
        self.dvAccum_CN_N = dvAccum_CN_N or torch.zeros(3)

    def reset(self) -> None:
        return dict(
            total_orbital_energy=torch.zeros(1),
            total_rotation_energy=torch.zeros(1),
            state_effector_rotation_energy=torch.zeros(1),
            state_effector_orbital_energy=torch.zeros(1),
            dvAccum_CN_B=self.dvAccum_CN_B,
            dvAccum_BN_B=self.dvAccum_BN_B,
            dvAccum_CN_N=self.dvAccum_CN_N,
        )
