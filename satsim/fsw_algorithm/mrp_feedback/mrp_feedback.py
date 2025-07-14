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
        k: float,
        ki: float,
        p: float,
        integral_limit: float,
        control_law_Type: int,
        inertia_spacecraft_point_b_in_body: torch.Tensor,
        reaction_wheels_inertia_wrt_spin: torch.Tensor,
        reaction_wheels_spin_axis: torch.Tensor,
        known_torque_point_b_in_body: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.ki = ki
        self.p = p
        self.integral_limit = integral_limit
        self.control_law_type = control_law_Type

        known_torque_point_b_in_body = torch.zeros(
            1
        ) if known_torque_point_b_in_body is None else known_torque_point_b_in_body

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
        if self.ki > 0:
            integral_sigma = integral_sigma + self.k * dt * sigma_BR
            integral_sigma = torch.clamp(
                integral_sigma,
                min=-self.integral_limit,
                max=self.integral_limit,
            )
            state_dict['integral_sigma'] = integral_sigma
            z = integral_sigma + torch.matmul(
                inertia_spacecraft_point_b_in_body,
                omega_BR_B.unsqueeze(-1),
            ).squeeze(-1)  # [b, 3]

        integral_feedback_output = z * self.ki * self.p
        attitude_control_torque = sigma_BR * self.k + omega_BR_B * self.p + integral_feedback_output

        temp1 = torch.matmul(
            inertia_spacecraft_point_b_in_body,
            omega_BN_B.unsqueeze(-1),
        ).squeeze(-1)

        temp1 = (reaction_wheels_inertia_wrt_spin * (torch.matmul(
            omega_BN_B.unsqueeze(-2),
            reaction_wheels_spin_axis,
        ) + wheel_speeds) * reaction_wheels_spin_axis).sum(-1) + temp1

        temp2 = omega_RN_B + z * self.ki if self.control_law_type == 0 else omega_BN_B
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
