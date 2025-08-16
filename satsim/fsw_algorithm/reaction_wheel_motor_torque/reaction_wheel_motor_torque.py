__all__ = ['ReactionWheelMotorTorque']
import torch

from satsim.architecture import Module, VoidStateDict


class ReactionWheelMotorTorque(Module[VoidStateDict]):

    def __init__(
        self,
        *args,
        control_axis: torch.Tensor,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.register_buffer(
            '_control_axis',
            control_axis,
            persistent=False,
        )

    @property
    def control_axis(self) -> torch.Tensor:
        #[3, num_axis]
        return self.get_buffer('_control_axis')

    def forward(
        self,
        state_dict: VoidStateDict,
        *args,
        torque_request_body: torch.Tensor,  # [3]
        reaction_wheel_spin_axis_in_body: torch.Tensor,
        **kwargs,
    ) -> tuple[VoidStateDict, tuple[torch.Tensor]]:
        num_axis = self.control_axis.size(-1)
        num_reaction_wheels = reaction_wheel_spin_axis_in_body.size(-1)
        assert num_reaction_wheels == num_axis <= 3

        torque_request_body = -torque_request_body  # [3]

        torque_axis = torch.matmul(
            torque_request_body.unsqueeze(-2),
            self.control_axis,
        ).squeeze(-2)  # [num_axis]
        CGs = torch.matmul(
            self.control_axis.transpose(-1, -2),
            reaction_wheel_spin_axis_in_body,
        )  # [num_axis, num_reaction_wheels]

        # TODO: gradiant may be broken here
        m33 = torch.matmul(CGs, CGs.transpose(-1, -2))  # [num_axis, num_axis]
        temp = torch.linalg.solve(m33, torque_axis.unsqueeze(-1)).transpose(
            -1, -2)  # [1, num_axis]

        motor_torque = torch.matmul(temp, CGs).squeeze(-2)

        return state_dict, (motor_torque, )
