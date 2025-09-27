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
        control_law_type: torch.Tensor | None = None,
        known_torque_point_b_in_body: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        control_law_type = torch.zeros(
            1,
            dtype=torch.bool) if control_law_type is None else control_law_type

        known_torque_point_b_in_body = torch.zeros(
            3
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

        integral_sigma = (integral_sigma +
                          self.k.unsqueeze(-1) * dt * sigma_BR)

        integral_limit = self.integral_limit.unsqueeze(-1)
        clamp_mask = (torch.abs(integral_sigma) > integral_limit)
        integral_sigma = torch.where(
            clamp_mask,
            torch.copysign(integral_limit, integral_sigma),
            integral_sigma,
        )

        state_dict['integral_sigma'] = integral_sigma

        attitude_error_measure = integral_sigma + torch.einsum(
            '...ij, ...j -> ...i',
            inertia_spacecraft_point_b_in_body,
            omega_BR_B,
        )

        integral_feedback_output = (attitude_error_measure *
                                    self.ki.unsqueeze(-1) *
                                    self.p.unsqueeze(-1))  # v3_5
        attitude_control_torque = (sigma_BR * self.k.unsqueeze(-1) +
                                   omega_BR_B * self.p.unsqueeze(-1) +
                                   integral_feedback_output)  # Lr

        angular_momentum_BN_B = torch.einsum(
            '...ij,...j -> ...i',
            inertia_spacecraft_point_b_in_body,
            omega_BN_B,
        )
        angular_momentum = ((reaction_wheels_inertia_wrt_spin * (torch.einsum(
            '...i,...ij->j',
            omega_BN_B,
            reaction_wheels_spin_axis,
        ) + wheel_speeds) * reaction_wheels_spin_axis).sum(-1) +
                            angular_momentum_BN_B)  # v3_6

        temp2 = torch.where(
            self.control_law_type,
            omega_BN_B,
            omega_RN_B + attitude_error_measure * self.ki.unsqueeze(-1),
        )  # v3_8
        attitude_control_torque = attitude_control_torque + torch.cross(
            angular_momentum,
            temp2,
            dim=-1,
        )
        attitude_control_torque = attitude_control_torque + torch.einsum(
            '...ij,...j -> ...i',
            inertia_spacecraft_point_b_in_body,
            (omega_BN_B.cross(omega_RN_B, dim=-1) - domega_RN_B),
        ) + self.known_torque_point_b_in_body

        return state_dict, (
            -attitude_control_torque,
            -integral_feedback_output,
        )
