__all__ = [
    'LocationPointing',
    'LocationPointingStateDict',
    'LocationPointingOutput',
]

from typing import NamedTuple, NotRequired, TypedDict

import einops
import torch
import torch.nn.functional as F
from torch.autograd import Function

from satsim.architecture import Module, constants
from satsim.utils import add_mrp, mrp_to_rotation_matrix


class SafeAcos(Function):

    def forward(
        ctx,
        input: torch.Tensor,
        grad_dump: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(input, grad_dump)
        return torch.acos(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input, grad_dump) = ctx.saved_tensors
        grad_input = -1.0 / torch.sqrt(1 - input**2) * grad_output
        grad_input = torch.where(
            grad_dump,
            0.,
            grad_input,
        )
        return grad_input, None


class LocationPointingStateDict(TypedDict):
    attitude_BR_old: NotRequired[torch.Tensor]
    # [3] / [batch, ..., 3]


class LocationPointingOutput(NamedTuple):
    attitude_BR: torch.Tensor
    angular_velocity_BR_B: torch.Tensor
    angular_velocity_RN_B: torch.Tensor
    attitude_RN: torch.Tensor
    angular_velocity_RN_N: torch.Tensor


class LocationPointing(Module[LocationPointingStateDict]):

    def __init__(
        self,
        *args,
        pointing_direction_B_B: torch.Tensor,  # [b, ..., 3]
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.init = True

        pointing_direction_B_B_norm = torch.norm(pointing_direction_B_B,
                                                 dim=-1)
        if torch.any(
                pointing_direction_B_B_norm < constants.PARALLEL_TOLERANCE):
            print(
                f"locationPoint: vector p_hat_B is not setup as a unit vector. Min norm: {pointing_direction_B_B_norm.min().item()}"
            )
        self.register_buffer(
            '_pointing_direction_B_B',
            pointing_direction_B_B,
            persistent=False,
        )

        v1 = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(0)  # [1, 3]
        aux_perp_vector_180_B_B = torch.cross(pointing_direction_B_B,
                                              v1,
                                              dim=-1)  # [b, ..., 3]

        e_norm = torch.norm(aux_perp_vector_180_B_B, dim=-1)
        mask = e_norm < constants.PARALLEL_TOLERANCE
        if torch.any(mask):
            v1 = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)  # [1, 3]
            aux_perp_vector_180_B_B = torch.where(
                mask.unsqueeze(-1),  # [b, ..., 1]
                torch.cross(pointing_direction_B_B, v1, dim=-1),
                aux_perp_vector_180_B_B,
            )

        self.register_buffer(
            '_aux_perp_vector_180_B_B',
            aux_perp_vector_180_B_B,
            persistent=False,
        )

    @property
    def pointing_direction_B_B(self) -> torch.Tensor:
        return self.get_buffer('_pointing_direction_B_B')

    @property
    def aux_perp_vector_180_B_B(self) -> torch.Tensor:
        return self.get_buffer('_aux_perp_vector_180_B_B')

    def forward(
        self,
        state_dict: LocationPointingStateDict,
        *args,
        position_LN_N: torch.Tensor,
        position_BN_N: torch.Tensor,
        attitude_BN: torch.Tensor,
        angular_velocity_BN_B: torch.Tensor,
        **kwargs,
    ) -> tuple[LocationPointingStateDict, LocationPointingOutput]:
        """
        Computes the attitude guidance and reference messages for location pointing.
        Args:
            state_dict (LocationPointingStateDict): Dictionary containing the previous state, including "sigma_BR_old".
            *args: Additional positional arguments (unused).
            position_LN_N (torch.Tensor): Target location position in the inertial frame, shape [..., 3].
            position_BN_N (torch.Tensor): Spacecraft position in the inertial frame, shape [..., 3].
            attitude_BN (torch.Tensor): Spacecraft attitude represented as Modified Rodrigues Parameters (MRP), shape [..., 3].
            angular_velocity_BN_B (torch.Tensor): Spacecraft angular velocity in the inertial frame, shape [..., 3].
            **kwargs: Additional keyword arguments (unused).
        Returns:
            tuple:
                - updated_state (dict): Updated state dictionary containing "sigma_BR_old" and "aux_perp_vector_180_B_B".
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
        position_LB_N = position_LN_N - position_BN_N  # [b, ..., 3]

        # principle rotation angle to point pHat at location
        direction_cosine_matrix_BN = mrp_to_rotation_matrix(
            attitude_BN)  # [b, ..., 3, 3]
        position_LB_B = torch.einsum("...ij,...j->...i",
                                     direction_cosine_matrix_BN, position_LB_N)
        position_LB_B_unit = F.normalize(position_LB_B, dim=-1)

        dum1 = torch.sum(self.pointing_direction_B_B * position_LB_B_unit,
                         dim=-1)
        dum1 = torch.clamp(dum1, -1.0, 1.0)
        angle_error = torch.acos(dum1)

        non_parallel_mask = angle_error > constants.PARALLEL_TOLERANCE
        safe_angle_error = SafeAcos.apply(dum1, ~non_parallel_mask)

        # calculate sigma_BR
        attitude_BR = torch.zeros_like(attitude_BN)  # [b, ..., 3]

        if torch.any(non_parallel_mask):
            near_180_mask = (torch.pi -
                             safe_angle_error) < constants.PARALLEL_TOLERANCE

            cross_product = torch.cross(
                self.pointing_direction_B_B,
                position_LB_B_unit,
                dim=-1,
            )
            safe_cross_product = torch.where(
                (non_parallel_mask & ~near_180_mask).unsqueeze(-1),  # [b, ...]
                cross_product,
                torch.zeros_like(position_LB_B_unit),
            )
            rotation_axis = torch.where(
                near_180_mask.unsqueeze(-1),  # [b, ..., 1]
                self.aux_perp_vector_180_B_B,
                safe_cross_product,
            )
            rotation_axis = F.normalize(rotation_axis, dim=-1)
            attitude_BR = torch.where(
                non_parallel_mask.unsqueeze(-1),  # [b, ..., 1]
                -torch.tan(safe_angle_error.unsqueeze(-1) / 4) * rotation_axis,
                attitude_BR,
            )

        # compute sigma_RN
        attitude_RN = add_mrp(
            attitude_BN,  # N in B
            -attitude_BR,
        )

        angular_velocity_BR_B = torch.zeros_like(angular_velocity_BN_B)
        if 'attitude_BR_old' in state_dict:
            attitude_BR_old = state_dict['attitude_BR_old']
            difference = attitude_BR - attitude_BR_old
            attitude_BR_dot = difference / self._timer.dt
            binv = _binv_mrp(attitude_BR)
            attitude_BR_dot = 4 * attitude_BR_dot
            angular_velocity_BR_B = torch.einsum("...ij,...j->...i", binv,
                                                 attitude_BR_dot)
        state_dict['attitude_BR_old'] = attitude_BR

        angular_velocity_RN_B = angular_velocity_BN_B - angular_velocity_BR_B
        angular_velocity_RN_N = torch.einsum(
            "...ji,...j->...i",
            direction_cosine_matrix_BN,
            angular_velocity_RN_B,
        )
        return (
            state_dict,
            LocationPointingOutput(
                attitude_BR=attitude_BR,
                angular_velocity_BR_B=angular_velocity_BR_B,
                angular_velocity_RN_B=angular_velocity_RN_B,
                attitude_RN=attitude_RN,
                angular_velocity_RN_N=angular_velocity_RN_N,
            ),
        )


def _binv_mrp(q: torch.Tensor) -> torch.Tensor:
    s2 = torch.sum(q**2, dim=-1)

    q0, q1, q2 = q.unbind(-1)

    row0 = torch.stack(
        [1 - s2 + 2 * q0**2, 2 * (q0 * q1 + q2), 2 * (q0 * q2 - q1)], dim=-1)
    row1 = torch.stack(
        [2 * (q1 * q0 - q2), 1 - s2 + 2 * q1**2, 2 * (q1 * q2 + q0)], dim=-1)
    row2 = torch.stack(
        [2 * (q2 * q0 + q1), 2 * (q2 * q1 - q0), 1 - s2 + 2 * q2**2], dim=-1)
    binv = torch.stack([row0, row1, row2], dim=-2)

    scale = 1.0 / ((1 + s2)**2)
    scale = einops.repeat(scale, '... -> ... 3 3')
    binv = binv * scale

    return binv
