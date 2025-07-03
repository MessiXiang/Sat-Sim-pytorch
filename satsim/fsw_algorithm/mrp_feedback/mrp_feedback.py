__all__ = ['MRPFeedbackStateDict', 'MRPFeedback']
from typing import TypedDict

import torch

from satsim.architecture import Module
from .operator import softclamp


class MRPFeedbackStateDict(TypedDict):
    integral_sigma: torch.Tensor
    known_torque_point_b_body: torch.Tensor
    inertia_spacecraft_point_b_body: torch.Tensor


class MRPFeedback(Module[MRPFeedbackStateDict]):

    def __init__(
        self,
        *args,
        k: float,
        ki: float,
        p: float,
        integral_limit: float,
        control_law_Type: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.ki = ki
        self.p = p
        self.integral_limit = integral_limit
        self.control_law_type = control_law_Type

    def reset(self) -> MRPFeedbackStateDict:
        return dict(
            # z=torch.zeros(3),
            integral_sigma=torch.zeros(3),
            known_torque_point_b_body=torch.zeros(3),
            inertia_spacecraft_point_b_body=torch.zeros(3, 3),
        )

    def forward(
        self,
        state_dict: MRPFeedbackStateDict,
        *args,
        sigma_BR: torch.Tensor,
        omega_BR_B: torch.Tensor,
        omega_RN_B: torch.Tensor,
        domega_RN_B: torch.Tensor,
        wheel_speeds: torch.Tensor,
        gsHat_B: torch.Tensor,
        Js: torch.Tensor,
        **kwargs,
    ) -> tuple[
            MRPFeedbackStateDict,
            tuple[
                torch.Tensor,
                torch.Tensor,
            ],
    ]:
        integral_sigma = state_dict['integral_sigma']
        inertia_spacecraft_point_b_body = state_dict[
            'inertia_spacecraft_point_b_body']
        known_torque_point_b_body = state_dict['known_torque_point_b_body']
        num_reaction_wheels = Js.size(-1) if Js else 0

        dt = 0. if self._timer.time <= self._timer.dt else self._timer.dt

        omega_BN_B = omega_BR_B + omega_RN_B
        z = torch.zeros_like(omega_BN_B)
        if self.ki > 0:
            integral_sigma = integral_sigma + self.k * dt * sigma_BR
            integral_sigma = softclamp(
                integral_sigma,
                min=-self.integral_limit,
                max=self.integral_limit,
            )
            state_dict['integral_sigma'] = integral_sigma
            z = integral_sigma + inertia_spacecraft_point_b_body @ omega_BR_B

        integral_feedback_output = z * self.ki * self.p
        attitude_control_torque = sigma_BR * self.k + omega_BR_B * self.p + integral_feedback_output

        temp1 = inertia_spacecraft_point_b_body @ omega_BN_B

        if num_reaction_wheels > 0:
            temp1 = (Js * (omega_BN_B @ gsHat_B + wheel_speeds) *
                     gsHat_B).sum(-1) + temp1

        temp2 = omega_RN_B + z * self.ki if self.control_law_type == 0 else omega_BN_B
        attitude_control_torque = attitude_control_torque + temp1.cross(temp2)

        attitude_control_torque = attitude_control_torque + \
            inertia_spacecraft_point_b_body @ (
            omega_BN_B.cross(omega_RN_B) - domega_RN_B) + \
            known_torque_point_b_body

        return state_dict, (-attitude_control_torque, integral_feedback_output)
