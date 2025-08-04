__all__ = ['MRPFeedbackStateDict', 'MRPFeedback']
from typing import TypedDict

import torch

from satsim.architecture import Module


class MRPFeedbackStateDict(TypedDict):
    integral_sigma: torch.Tensor


class MRPFeedback(Module[MRPFeedbackStateDict]):

    def __init__(
        self,
        *args,
        k: torch.Tensor,
        ki: torch.Tensor,
        p: torch.Tensor,
        integral_limit: torch.Tensor,
        control_law_type: torch.BoolTensor,
        inertia_spacecraft_point_b_in_body: torch.Tensor,
        reaction_wheels_inertia_wrt_spin: torch.Tensor,
        reaction_wheels_spin_axis: torch.Tensor,
        known_torque_point_b_in_body: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.control_law_type = control_law_type

        known_torque_point_b_in_body = torch.zeros(
            1
        ) if known_torque_point_b_in_body is None else known_torque_point_b_in_body

        self.register_buffer('k', k, persistent=False)
        self.register_buffer('ki', ki, persistent=False)
        self.register_buffer('p', p, persistent=False)
        self.register_buffer(
            'integral_limit',
            integral_limit,
            persistent=False,
        )
        self.register_buffer(
            'control_law_type',
            control_law_type,
            persistent=False,
        )
        self.register_buffer(
            'known_torque_point_b_in_body',
            known_torque_point_b_in_body,
            persistent=False,
        )
        self.register_buffer(
            'inertia_spacecraft_point_b_in_body',
            inertia_spacecraft_point_b_in_body,
            persistent=False,
        )
        self.register_buffer(
            'reaction_wheels_inertia_wrt_spin',
            reaction_wheels_inertia_wrt_spin,
            persistent=False,
        )
        self.register_buffer(
            'reaction_wheels_spin_axis',
            reaction_wheels_spin_axis,
            persistent=False,
        )

    def reset(self) -> MRPFeedbackStateDict:
        return dict(integral_sigma=torch.zeros(3), )

    def forward(
        self,
        state_dict: MRPFeedbackStateDict,
        *args,
        sigma_BR: torch.Tensor,
        omega_BR_B: torch.Tensor,
        omega_RN_B: torch.Tensor,
        domega_RN_B: torch.Tensor,
        wheel_speeds: torch.Tensor,
        **kwargs,
    ) -> tuple[
            MRPFeedbackStateDict,
            tuple[
                torch.Tensor,
                torch.Tensor,
            ],
    ]:

        integral_sigma = state_dict['integral_sigma']
        k = self.get_buffer('k')
        ki = self.get_buffer('ki')
        p = self.get_buffer('p')
        integral_limit = self.get_buffer('integral_limit')
        control_law_type = self.get_buffer('control_law_type')
        inertia_spacecraft_point_b_in_body = self.get_buffer(
            'inertia_spacecraft_point_b_in_body')
        known_torque_point_b_in_body = self.get_buffer(
            'known_torque_point_b_in_body')
        reaction_wheels_inertia_wrt_spin = self.get_buffer(
            'reaction_wheels_inertia_wrt_spin')
        reaction_wheels_spin_axis = self.get_buffer(
            'reaction_wheels_spin_axis')

        dt = 0. if self._timer.time <= self._timer.dt else self._timer.dt

        omega_BN_B = omega_BR_B + omega_RN_B
        z = torch.zeros_like(omega_BN_B)

        integral_mask = ki > 0
        integral_sigma = torch.where(integral_mask,
                                     integral_sigma + k * dt * sigma_BR,
                                     integral_sigma)
        clamp_mask = integral_mask & (torch.abs(integral_sigma)
                                      > integral_limit)
        integral_sigma = torch.where(
            clamp_mask,
            torch.copysign(integral_limit, integral_sigma),
            integral_sigma,
        )

        state_dict['integral_sigma'] = integral_sigma

        z = torch.where(
            integral_mask,
            integral_sigma + torch.matmul(
                inertia_spacecraft_point_b_in_body,
                omega_BR_B.unsqueeze(-1),
            ).squeeze(-1),  # [b, 3]
            z,
        )

        integral_feedback_output = z * ki * p  # v3_5
        attitude_control_torque = sigma_BR * k + omega_BR_B * p + integral_feedback_output

        temp1 = torch.matmul(
            inertia_spacecraft_point_b_in_body,
            omega_BN_B.unsqueeze(-1),
        ).squeeze(-1)

        temp1 = (reaction_wheels_inertia_wrt_spin * (torch.matmul(
            omega_BN_B.unsqueeze(-2),
            reaction_wheels_spin_axis,
        ) + wheel_speeds) * reaction_wheels_spin_axis).sum(-1) + temp1

        temp2 = torch.where(
            control_law_type,
            omega_BN_B,
            omega_RN_B + z * ki,
        )
        attitude_control_torque = attitude_control_torque + torch.cross(
            temp1,
            temp2,
            dim=-1,
        )

        attitude_control_torque = attitude_control_torque + torch.matmul(
            inertia_spacecraft_point_b_in_body,
            omega_BN_B.cross(omega_RN_B, dim=-1) - domega_RN_B,
        ) + known_torque_point_b_in_body

        return state_dict, (-attitude_control_torque, integral_feedback_output)
