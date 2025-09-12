__all__ = ["AttTrackingErrorStateDict", "AttTrackingError"]

from typing import TypedDict

import torch
from satsim.architecture import Module
from satsim.utils import mrp_to_rotation_matrix, add_mrp, sub_mrp


class AttTrackingErrorStateDict(TypedDict):
    pass


class AttTrackingError(Module[AttTrackingErrorStateDict]):

    def __init__(self, *args, attitude_R0R: torch.Tensor, **kwargs):

        super().__init__(*args, **kwargs)
        self.register_buffer(
            '_attitude_R0R',
            attitude_R0R,
            persistent=False,
        )

    @property
    def attitude_R0R(self) -> torch.Tensor:
        return self.get_buffer('_attitude_R0R')

    def forward(
        self,
        state_dict: AttTrackingErrorStateDict | None,
        attitude_RN: torch.Tensor,
        angular_velocity_RN_N: torch.Tensor,
        angular_acceleration_RN_N: torch.Tensor,
        attitude_BN: torch.Tensor,
        angular_velocity_BN_B: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[AttTrackingErrorStateDict | None, tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
    ]]:

        attitude_RR0 = -self.attitude_R0R
        attitude_RN = add_mrp(attitude_RN, attitude_RR0)

        attitude_BR = sub_mrp(attitude_BN, attitude_RN)

        direction_cosine_matrix_BN = mrp_to_rotation_matrix(attitude_BN)

        angular_velocity_RN_B = torch.einsum(
            '...ij,...j->...i',
            direction_cosine_matrix_BN,
            angular_velocity_RN_N,
        )
        angular_acceleration_RN_B = torch.einsum(
            '...ij,...j->...i',
            direction_cosine_matrix_BN,
            angular_acceleration_RN_N,
        )

        angular_velocity_BR_B = angular_velocity_BN_B - angular_velocity_RN_B

        return state_dict, (
            attitude_BR,
            angular_velocity_BR_B,
            angular_velocity_RN_B,
            angular_acceleration_RN_B,
        )
