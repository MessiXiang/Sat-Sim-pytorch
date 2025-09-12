__all__ = [
    'SimpleNavigator',
    'SimpleNavigatorStateDict',
]

from typing import TypedDict

import torch

from satsim.architecture import Module
from satsim.utils import mrp_to_rotation_matrix


class SimpleNavigatorStateDict(TypedDict):
    pass


# cross_correlation_for_translation and cross_correlation_for_attitude are deserted, so does propagate matrix
# noise_process_covariance_matrix is deserted
# noise walk bounds is deserted


# After modification, this module simply calculate sun position in body frame
# Ergo, if sun_position is not needed, this module can be deserted
class SimpleNavigator(Module[SimpleNavigatorStateDict]):

    def forward(
        self,
        state_dict: SimpleNavigatorStateDict,
        *args,
        position_BN_N: torch.
        Tensor,  # position of spacecraft in inertial frame [n_sc,3]
        attitude_BN: torch.Tensor,  # MRP altitude of spacecraft [n_sc, 3]
        position_SN_N: torch.Tensor,  # position of sun in inertial frame [3]
        **kwargs,
    ) -> tuple[SimpleNavigatorStateDict, tuple[torch.Tensor]]:

        # Calculate sun position in body frame
        position_SB_N = position_SN_N - position_BN_N
        position_SB_N_unit = torch.nn.functional.normalize(position_SB_N,
                                                           dim=-1)
        direction_cosine_matrix_BN = mrp_to_rotation_matrix(attitude_BN)
        position_SB_B = torch.einsum(
            '...ij,...j->...i',
            direction_cosine_matrix_BN,
            position_SB_N_unit,
        )

        return state_dict, (position_SB_B, )
