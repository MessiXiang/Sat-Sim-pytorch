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
        torque_request_body = -torque_request_body  # [3]
        torque_axis = torch.einsum(
            '...i, ... ij-> ...j',
            torque_request_body,
            self.control_axis,
        )  # [num_axis]

        CGs = torch.einsum(
            '...ij, ...ik -> ...jk',
            self.control_axis,
            reaction_wheel_spin_axis_in_body,
        )  # [num_axis, num_reaction_wheels]

        # TODO: gradiant may be broken here
        m33 = torch.einsum(
            '...ij, ...kj-> ...ik',
            CGs,
            CGs,
        )  # [num_axis, num_axis]
        temp: torch.Tensor = torch.linalg.solve(
            m33, torque_axis.unsqueeze(-1)).transpose(-1, -2)  # [1, num_axis]

        motor_torque = torch.einsum(
            '...ij, ...jk-> ...ik',
            temp,
            CGs,
        ).squeeze(-2)

        return state_dict, (motor_torque, )
