__all__ = ["AttTrackingErrorStateDict", "AttTrackingError"]

from typing import TypedDict

import torch
from satsim.architecture import Module
from satsim.utils import to_rotation_matrix, add_mrp, sub_mrp


class AttTrackingErrorStateDict(TypedDict):
    pass


class AttTrackingError(Module[AttTrackingErrorStateDict]):

    def __init__(self, *args, sigma_R0R: torch.Tensor | None = None, **kwargs):

        super().__init__(*args, **kwargs)
        sigma_R0R = torch.tensor(
            [0.01, 0.05, -0.55],
            dtype=torch.float32) if sigma_R0R is None else sigma_R0R

        self.register_buffer(
            '_sigma_R0R',
            sigma_R0R,
            persistent=False,
        )

    @property
    def sigma_R0R(self) -> torch.Tensor:
        return self.get_buffer('_sigma_R0R')

    def forward(
        self,
        state_dict: AttTrackingErrorStateDict | None,
        sigma_R0N: torch.Tensor,
        omega_RN_N: torch.Tensor,
        domega_RN_N: torch.Tensor,
        sigma_BN: torch.Tensor,
        omega_BN_B: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[AttTrackingErrorStateDict | None, tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
    ]]:

        sigma_RR0 = -self.sigma_R0R
        sigma_RN = add_mrp(sigma_R0N, sigma_RR0)

        sigma_BR = sub_mrp(sigma_BN, sigma_RN)

        dcm_BN = to_rotation_matrix(sigma_BN)

        omega_RN_B = torch.matmul(dcm_BN, omega_RN_N.unsqueeze(-1)).squeeze(-1)
        domega_RN_B = torch.matmul(dcm_BN,
                                   domega_RN_N.unsqueeze(-1)).squeeze(-1)

        omega_BR_B = omega_BN_B - omega_RN_B

        return state_dict, (sigma_BR, omega_BR_B, omega_RN_B, domega_RN_B)
