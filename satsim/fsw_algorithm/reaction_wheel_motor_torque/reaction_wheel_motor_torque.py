__all__ = ['ReactionWheelMotorTorqueStateDict', 'ReactionWheelMotorTorque']
import torch

from satsim.architecture import Module, VoidStateDict


class ReactionWheelMotorTorque(Module[VoidStateDict]):

    def __init__(
        self,
        *args,
        control_axis: torch.Tensor,
        reaction_wheel_spin_axis_in_body: torch.Tensor,
        **kwargs,
    ):
        num_axis = control_axis.size(-1)
        num_reaction_wheels = reaction_wheel_spin_axis_in_body.size(-1)
        assert num_reaction_wheels > num_axis and num_axis <= 3
        self.register_buffer(
            'control_axis',
            control_axis,
            persistent=False,
        )
        self.register_buffer(
            'reaction_wheel_spin_axis_in_body',
            reaction_wheel_spin_axis_in_body,
            persistent=False,
        )

    def forward(
        self,
        *args,
        torque_request_body: torch.Tensor,  # [3]
        **kwargs,
    ) -> tuple[VoidStateDict, tuple[torch.Tensor]]:
        control_axis = self.get_buffer('control_axis')  # [3, num_axis]
        reaction_wheel_spin_axis_in_body = self.get_buffer(
            'reaction_wheel_spin_axis_in_body')  # [3, num_reaction_wheel]

        torque_request_body = -torque_request_body  # [3]

        torque_axis = torch.matmul(
            torque_request_body.unsqueeze(-2),
            control_axis,
        ).squeeze(-2)  # [num_axis]
        CGs = torch.zeros_like(reaction_wheel_spin_axis_in_body)
        CGs = torch.matmul(
            control_axis.transpose(-1, -2),
            reaction_wheel_spin_axis_in_body,
        )  # [num_axis, num_reaction_wheels]

        # TODO: gradiant may be broken here
        m33 = torch.matmul(CGs, CGs.transpose(-1, -2))  # [num_axis, num_axis]
        temp = torch.linalg.solve(m33, torque_axis.unsqueeze(-1)).transpose(
            -1, -2)  # [1, num_axis]

        motor_torque = torch.matmul(temp, CGs).squeeze(-2)

        return dict(), (motor_torque, )
