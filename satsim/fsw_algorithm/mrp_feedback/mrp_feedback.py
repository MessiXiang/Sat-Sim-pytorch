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
        control_law_type: torch.BoolTensor | None = None,
        known_torque_point_b_in_body: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        control_law_type = torch.zeros(
            1) if control_law_type is None else control_law_type

        known_torque_point_b_in_body = torch.zeros(
            1
        ) if known_torque_point_b_in_body is None else known_torque_point_b_in_body

        self.register_buffer('_k', k, persistent=False)
        self.register_buffer('_ki', ki, persistent=False)
        self.register_buffer('_p', p, persistent=False)
        self.register_buffer(
            '_integral_limit',
            integral_limit,
            persistent=False,
        )
        self.register_buffer(
            '_control_law_type',
            control_law_type,
            persistent=False,
        )
        self.register_buffer(
            '_known_torque_point_b_in_body',
            known_torque_point_b_in_body,
            persistent=False,
        )

    @property
    def k(self) -> torch.Tensor:
        return self.get_buffer('_k')

    @property
    def ki(self) -> torch.Tensor:
        return self.get_buffer('_ki')

    @property
    def p(self) -> torch.Tensor:
        return self.get_buffer('_p')

    @property
    def integral_limit(self) -> torch.Tensor:
        return self.get_buffer('_integral_limit')

    @property
    def known_torque_point_b_in_body(self) -> torch.Tensor:
        return self.get_buffer('_known_torque_point_b_in_body')

    @property
    def control_law_type(self) -> torch.Tensor:
        return self.get_buffer('_control_law_type')

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
        inertia_spacecraft_point_b_in_body: torch.Tensor,
        reaction_wheels_inertia_wrt_spin: torch.Tensor,
        reaction_wheels_spin_axis: torch.Tensor,
        **kwargs,
    ) -> tuple[
            MRPFeedbackStateDict,
            tuple[
                torch.Tensor,
                torch.Tensor,
            ],
    ]:
        integral_sigma = state_dict['integral_sigma']

        dt = 0. if self._timer.time <= self._timer.dt else self._timer.dt

        omega_BN_B = omega_BR_B + omega_RN_B
        z = torch.zeros_like(omega_BN_B)

        integral_mask = self.ki > 0
        integral_sigma = torch.where(integral_mask,
                                     integral_sigma + self.k * dt * sigma_BR,
                                     integral_sigma)
        clamp_mask = integral_mask & (torch.abs(integral_sigma)
                                      > self.integral_limit)
        integral_sigma = torch.where(
            clamp_mask,
            torch.copysign(self.integral_limit, integral_sigma),
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

        integral_feedback_output = z * self.ki * self.p  # v3_5
        attitude_control_torque = sigma_BR * self.k + omega_BR_B * self.p + integral_feedback_output

        temp1 = torch.matmul(
            inertia_spacecraft_point_b_in_body,
            omega_BN_B.unsqueeze(-1),
        ).squeeze(-1)

        temp1 = (reaction_wheels_inertia_wrt_spin * (torch.matmul(
            omega_BN_B.unsqueeze(-2),
            reaction_wheels_spin_axis,
        ) + wheel_speeds) * reaction_wheels_spin_axis).sum(-1) + temp1

        temp2 = torch.where(
            self.control_law_type,
            omega_BN_B,
            omega_RN_B + z * self.ki,
        )
        attitude_control_torque = attitude_control_torque + torch.cross(
            temp1,
            temp2,
            dim=-1,
        )

        attitude_control_torque = attitude_control_torque + torch.matmul(
            inertia_spacecraft_point_b_in_body,
            omega_BN_B.cross(omega_RN_B, dim=-1) - domega_RN_B,
        ) + self.known_torque_point_b_in_body

        return state_dict, (-attitude_control_torque, integral_feedback_output)
