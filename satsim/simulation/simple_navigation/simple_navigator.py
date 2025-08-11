__all__ = [
    'SimpleNavigator',
    'SimpleNavigatorStateDict',
]

from typing import TypedDict

import torch

from satsim.architecture import Module
from satsim.utils import to_rotation_matrix


class SimpleNavigatorStateDict(TypedDict):
    pass


# cross_correlation_for_translation and cross_correlation_for_attitude are deserted, so does propagate matrix
# noise_process_covariance_matrix is deserted
# noise walk bounds is deserted


# After modification, this module simply calculate sun position in body frame
# Ergo, if sun_position is not provided, this module can be deserted
class SimpleNavigator(Module[SimpleNavigatorStateDict]):

    def forward(
        self,
        state_dict: SimpleNavigatorStateDict,
        *args,
        position_in_inertial: torch.Tensor,
        mrp_attitude_in_inertial: torch.Tensor,
        sun_position_in_inertial: torch.Tensor,
        **kwargs,
    ) -> tuple[SimpleNavigatorStateDict, tuple[torch.Tensor]]:
        # Calculate sun position in body frame
        sc2SunInrtl = sun_position_in_inertial - position_in_inertial
        sc2SunInrtl = torch.nn.functional.normalize(sc2SunInrtl, dim=-1)
        dcm_BN = to_rotation_matrix(mrp_attitude_in_inertial)
        sun_direction_in_body = torch.matmul(
            dcm_BN, sc2SunInrtl.unsqueeze(-1)).squeeze(-1)

        return state_dict, (sun_direction_in_body, )
