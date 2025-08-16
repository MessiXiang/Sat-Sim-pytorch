__all__ = [
    'LocationPointing',
    'LocationPointingStateDict',
    'LocationPointingOutput',
]

from typing import NamedTuple, NotRequired, TypedDict

import torch
import torch.nn.functional as F

from satsim.architecture import Module, constants
from satsim.utils import add_mrp, to_rotation_matrix


class LocationPointingStateDict(TypedDict):
    attitude_reference_in_body_old: NotRequired[torch.Tensor]
    # [3] / [batch, ..., 3]


class LocationPointingOutput(NamedTuple):
    attitude_reference_in_body: torch.Tensor
    angular_velocity_reference_wrt_body_in_body: torch.Tensor
    attitude_inertial_in_reference: torch.Tensor
    angular_velocity_reference_wrt_inertial_in_inertial: torch.Tensor


class LocationPointing(Module[LocationPointingStateDict]):

    def __init__(
        self,
        *args,
        pointing_direction: torch.Tensor | None = None,  # [b, ..., 3]
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.init = True

        p_norm = torch.norm(pointing_direction, dim=-1)
        if torch.any(p_norm < constants.PARALLEL_TOLERANCE):
            print(
                f"locationPoint: vector p_hat_B is not setup as a unit vector. Min norm: {p_norm.min().item}"
            )
        self.register_buffer(
            '_pointing_direction',
            pointing_direction,
            persistent=False,
        )

        v1 = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(0)  # [1, 3]
        e_hat_180_B = torch.cross(pointing_direction, v1,
                                  dim=-1)  # [b, ..., 3]

        e_norm = torch.norm(e_hat_180_B, dim=-1)
        mask = e_norm < constants.PARALLEL_TOLERANCE
        if torch.any(mask):
            v1 = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)  # [1, 3]
            e_hat_180_B = torch.where(
                mask.unsqueeze(-1),  # [b, ..., 1]
                torch.cross(pointing_direction, v1, dim=-1),
                e_hat_180_B,
            )

        self.register_buffer(
            '_aux_perp_vector_near_180',
            pointing_direction,
            persistent=False,
        )

    @property
    def pointing_direction(self) -> torch.Tensor:
        return self.get_buffer('_pointing_direction')

    @property
    def aux_perp_vector_near_180(self) -> torch.Tensor:
        return self.get_buffer('_aux_perp_vector_near_180')

    def forward(
        self,
        state_dict: LocationPointingStateDict,
        *args,
        target_position_in_inertial: torch.Tensor,
        spacecraft_position_in_inertial: torch.Tensor,
        spacecraft_attitude: torch.Tensor,
        spacecraft_angular_velocity_in_body: torch.Tensor,
        **kwargs,
    ) -> tuple[LocationPointingStateDict, LocationPointingOutput]:
        """
        Computes the attitude guidance and reference messages for location pointing.
        Args:
            state_dict (LocationPointingStateDict): Dictionary containing the previous state, including "sigma_BR_old".
            *args: Additional positional arguments (unused).
            target_position_in_inertial (torch.Tensor): Target location position in the inertial frame, shape [..., 3].
            spacecraft_position_in_inertial (torch.Tensor): Spacecraft position in the inertial frame, shape [..., 3].
            spacecraft_attitude (torch.Tensor): Spacecraft attitude represented as Modified Rodrigues Parameters (MRP), shape [..., 3].
            spacecraft_angular_velocity_in_inertial (torch.Tensor): Spacecraft angular velocity in the inertial frame, shape [..., 3].
            **kwargs: Additional keyword arguments (unused).
        Returns:
            tuple:
                - updated_state (dict): Updated state dictionary containing "sigma_BR_old" and "e_hat_180_B".
                - (att_guidance_message, att_reference_message) (tuple of dicts):
                    - att_guidance_message (dict): Contains "sigma_BR" (MRP error between body and reference) and "omega_BR_B" (angular velocity error in body frame).
                    - att_reference_message (dict): Contains "sigma_RN" (MRP error between reference and inertial) and "omega_RN_N" (reference angular velocity in inertial frame).
        Notes:
            - This method calculates the attitude error and reference for pointing the spacecraft at a target location.
            - Handles special cases for parallel and near-180-degree pointing directions.
            - Uses Modified Rodrigues Parameters (MRP) for attitude representation.
        """
        # R or reference frame is defined by sigma_BR. It means if new_sigma_BN = sigma_RB + sigma_BN, pointing vector point at the target

        # calculate r_LS_N
        target_position_wrt_body_in_inertial = target_position_in_inertial - spacecraft_position_in_inertial  # [b, ..., 3]

        # principle rotation angle to point pHat at location
        dcm_BN = to_rotation_matrix(spacecraft_attitude)  # [b, ..., 3, 3]
        target_position_in_body = torch.matmul(
            dcm_BN,
            target_position_wrt_body_in_inertial.unsqueeze(-1)).squeeze(-1)
        target_direction_in_body = target_position_in_body / torch.norm(
            target_position_in_body,
            dim=-1,
            keepdim=True,
        )  # [b, ..., 3]

        dum1 = torch.sum(self.pointing_direction * target_direction_in_body,
                         dim=-1)
        dum1 = torch.clamp(dum1, -1.0, 1.0)
        angle_error = torch.acos(dum1)

        # calculate sigma_BR
        attitude_reference_in_body = torch.zeros_like(
            spacecraft_attitude)  # [b, ..., 3]
        non_parallel_mask = angle_error > constants.PARALLEL_TOLERANCE

        if torch.any(non_parallel_mask):
            near_180_mask = (torch.pi -
                             angle_error) < constants.PARALLEL_TOLERANCE
            rotation_axis = torch.where(
                near_180_mask.unsqueeze(-1),  # [b, ..., 1]
                self.aux_perp_vector_near_180,
                torch.cross(self.pointing_direction,
                            target_direction_in_body,
                            dim=-1))
            rotation_axis = F.normalize(rotation_axis, dim=-1)
            attitude_reference_in_body = torch.where(
                non_parallel_mask.unsqueeze(-1),  # [b, ..., 1]
                -torch.tan(angle_error.unsqueeze(-1) / 4) * rotation_axis,
                attitude_reference_in_body,
            )

        # compute sigma_RN
        attitude_inertial_in_reference = add_mrp(
            spacecraft_attitude,  # N in B
            -attitude_reference_in_body,
        )

        angular_velocity_reference_wrt_body_in_body = torch.zeros_like(
            spacecraft_angular_velocity_in_body)
        if 'attitude_reference_in_body_old' in state_dict:
            attitude_reference_in_body_old = state_dict[
                'attitude_reference_in_body_old']
            difference = attitude_reference_in_body - attitude_reference_in_body_old
            sigma_dot_BR = difference / self._timer.dt
            binv = _binv_mrp(attitude_reference_in_body)
            sigma_dot_BR = 4 * sigma_dot_BR
            angular_velocity_reference_wrt_body_in_body = torch.matmul(
                binv, sigma_dot_BR.unsqueeze(-1)).squeeze(-1)

        angular_velocity_reference_wrt_inertial_in_inertial = torch.matmul(
            dcm_BN.transpose(-2, -1),
            (spacecraft_angular_velocity_in_body -
             angular_velocity_reference_wrt_body_in_body).unsqueeze(-1),
        ).squeeze(-1)

        return (
            LocationPointingStateDict(
                attitude_reference_in_body_old=attitude_reference_in_body),
            LocationPointingOutput(
                attitude_reference_in_body=attitude_reference_in_body,
                angular_velocity_reference_wrt_body_in_body=
                angular_velocity_reference_wrt_body_in_body,
                attitude_inertial_in_reference=attitude_inertial_in_reference,
                angular_velocity_reference_wrt_inertial_in_inertial=
                angular_velocity_reference_wrt_inertial_in_inertial,
            ),
        )


def _binv_mrp(q: torch.Tensor) -> torch.Tensor:
    s2 = torch.sum(q**2, dim=-1, keepdim=True)

    q0 = q[..., 0:1]
    q1 = q[..., 1:2]
    q2 = q[..., 2:3]

    batch_size = q.shape[:-1]
    binv = torch.zeros(*batch_size, 3, 3, device=q.device, dtype=q.dtype)

    # Compute each element of the matrix using MRP B inverse formula
    binv[..., 0, 0] = (1 - s2 + 2 * q0**2).squeeze(-1)
    binv[..., 0, 1] = (2 * (q0 * q1 + q2)).squeeze(-1)
    binv[..., 0, 2] = (2 * (q0 * q2 - q1)).squeeze(-1)

    binv[..., 1, 0] = (2 * (q1 * q0 - q2)).squeeze(-1)
    binv[..., 1, 1] = (1 - s2 + 2 * q1**2).squeeze(-1)
    binv[..., 1, 2] = (2 * (q1 * q2 + q0)).squeeze(-1)

    binv[..., 2, 0] = (2 * (q2 * q0 + q1)).squeeze(-1)
    binv[..., 2, 1] = (2 * (q2 * q1 - q0)).squeeze(-1)
    binv[..., 2, 2] = (1 - s2 + 2 * q2**2).squeeze(-1)

    scale = 1.0 / ((1 + s2)**2)
    binv = binv * scale.squeeze(-1).unsqueeze(-1).unsqueeze(-1)

    return binv
