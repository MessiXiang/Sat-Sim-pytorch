__all__ = [
    'SimpleNavigator',
    'SimpleNavigatorStateDict',
    'AttitudeData',
    'TranslationData',
]

from typing import TypedDict

import torch

from satsim.architecture import Module
from satsim.utils import to_rotation_matrix


class AttitudeData(TypedDict):
    mrp_attitude_in_inertial: torch.Tensor  # [3]  [-]    Current spacecraft attitude (MRPs) of body relative to inertial
    angular_velocity_in_inertial: torch.Tensor  # [3]  [r/s]  Current spacecraft angular velocity vector of body frame B relative to inertial frame N, in B frame components
    sun_position_in_body: torch.Tensor  # [3]  [m]    Current sun pointing vector in body frame


class TranslationData(TypedDict):
    position_in_inertial: torch.Tensor  # [3]  [m]    Current inertial spacecraft position vector in inertial frame N components
    velocity_in_inertial: torch.Tensor  # [3]  [m/s]  Current inertial velocity of the spacecraft in inertial frame N components
    total_accumulated_delta_velocity_in_inertial: torch.Tensor  # [3]  [m/s]  Total accumulated delta-velocity for s/c


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
        velocity_in_inertial: torch.Tensor,
        mrp_attitude_in_inertial: torch.Tensor,
        angular_velocity_in_inertial: torch.Tensor,
        total_accumulated_delta_velocity_in_inertial: torch.Tensor,
        sun_position_in_inertial: torch.Tensor,
        **kwargs,
    ) -> tuple[SimpleNavigatorStateDict, tuple[AttitudeData, TranslationData]]:
        # Calculate sun position in body frame
        sc2SunInrtl = sun_position_in_inertial - position_in_inertial
        sc2SunInrtl = torch.nn.functional.normalize(sc2SunInrtl, dim=-1)
        dcm_BN = to_rotation_matrix(mrp_attitude_in_inertial)
        sun_position_in_body = torch.matmul(
            dcm_BN, sc2SunInrtl.unsqueeze(-1)).squeeze(-1)

        estimated_translation_state = TranslationData(
            position_in_inertial=position_in_inertial,
            velocity_in_inertial=velocity_in_inertial,
            total_accumulated_delta_velocity_in_inertial=
            total_accumulated_delta_velocity_in_inertial,
        )

        estimated_attitude_state = AttitudeData(
            mrp_attitude_in_inertial=mrp_attitude_in_inertial,
            angular_velocity_in_inertial=angular_velocity_in_inertial,
            sun_position_in_body=sun_position_in_body,
        )

        return state_dict, (estimated_attitude_state,
                            estimated_translation_state)
