__all__ == ['ReactionWheelMotorTorqueStateDict', 'ReactionWheelMotorTorque']
from typing import TypedDict

import torch

from satsim.architecture import Module


class ReactionWheelMotorTorqueStateDict(TypedDict):
    control_axis: torch.Tensor


class ReactionWheelMotorTorque(Module[ReactionWheelMotorTorqueStateDict]):

    def reset(self):
        return dict(control_axis=torch.zeros(3, 3), )

    def forward(
        self,
        state_dict: ReactionWheelMotorTorqueStateDict,
        *args,
        torque_request_body: torch.Tensor,
        gsHat_B: torch.Tensor,
        **kwargs,
    ) -> tuple[ReactionWheelMotorTorqueStateDict, tuple[torch.Tensor]]:
        control_axis = state_dict['control_axis']
        num_axis = control_axis.size(-1)
        num_reaction_wheels = gsHat_B.size(-1)

        torque_request_body = -torque_request_body

        torque_axis = torch.zeros(3, device=gsHat_B.device)
        torque_axis[:num_axis] = torque_request_body @ control_axis
        CGs = torch.zeros(3, num_reaction_wheels, gsHat_B.device)
        CGs[:num_axis, :num_reaction_wheels] = control_axis.t() @ gsHat_B

        assert gsHat_B.size(-1) >= control_axis.size(-1)

        m33 = torch.eye(num_axis, num_axis, device=gsHat_B.device)
        m33[:num_axis, :num_axis] = CGs @ CGs.t()
        temp = m33.t() @ torque_axis

        motor_torque = temp @ CGs

        return state_dict, (motor_torque, )
