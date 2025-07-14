__all__ = ['MrpSteering', 'MrpSteeringDict']

from typing import TypedDict, Tuple
from satsim.architecture import Module
import torch


class MrpSteeringDict(TypedDict):
    pass


class MrpSteering(Module[MrpSteeringDict]):
    """
    parameters:
        K1                                    [rad/sec] Proportional gain applied to MRP errors
        K3                                    [rad/sec] Cubic gain applied to MRP error in steering saturation function
        omega_max                             [rad/sec] Maximum rate command of steering control
        ignore_outer_loop_feed_forward        []        Boolean flag indicating if outer feedforward term should be included
        attitude_errors_relative_to_reference [3]  [-]    Current attitude error estimate (MRPs) of B relative to R
    
    return:
        angular_velocity_relative_to_reference_in_body_frame     [3]  [r/s]   Desired body rate relative to R
        angular_acceleration_relative_to_reference_in_body_frame [3]  [r/s^2] Body-frame derivative of omega_body_relative_to_reference_in_body_frame
    """

    def __init__(
        self,
        *args,
        K1: torch.Tensor,
        K3: torch.Tensor,
        omega_max: torch.Tensor,
        ignore_outer_loop_feed_forward: torch.BoolTensor,
        attitude_errors_relative_to_reference: torch.Tensor,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self.register_buffer('K1', K1)
        self.register_buffer('K3', K3)
        self.register_buffer('omega_max', omega_max)
        self.register_buffer('ignore_outer_loop_feed_forward',
                             ignore_outer_loop_feed_forward)
        self.register_buffer('attitude_errors_relative_to_reference',
                             attitude_errors_relative_to_reference)

    def reset(self) -> MrpSteeringDict:
        return {}

    def forward(
            self, state_dict: MrpSteeringDict, *args, **kwargs
    ) -> tuple[MrpSteeringDict, Tuple[torch.Tensor, torch.Tensor]]:

        omega_body_relative_to_reference_in_body_frame, angular_acceleration_relative_to_reference_in_body_frame = self.mrp_steering_law(
        )
        return state_dict, (
            omega_body_relative_to_reference_in_body_frame,
            angular_acceleration_relative_to_reference_in_body_frame)

    def mrp_steering_law(self) -> Tuple[torch.Tensor, torch.Tensor]:
        K1 = self.get_buffer('K1')
        K3 = self.get_buffer('K3')
        omega_max = self.get_buffer('omega_max')
        ignore_outer_loop_feed_forward = self.get_buffer(
            'ignore_outer_loop_feed_forward')
        attitude_errors_relative_to_reference = self.get_buffer(
            'attitude_errors_relative_to_reference')

        sigma_cubed = attitude_errors_relative_to_reference**3
        inner_value = (torch.pi / 2) / omega_max * (
            K1 * attitude_errors_relative_to_reference + K3 * sigma_cubed)
        omega_ast = -torch.atan(inner_value) / (torch.pi / 2) * omega_max

        omega_ast_p = torch.zeros(3, dtype=torch.float64)
        if not ignore_outer_loop_feed_forward:
            B = BmatMRP(attitude_errors_relative_to_reference)
            sigma_p = 0.25 * torch.mv(B, omega_ast)

            numerator = 3 * K3 * (attitude_errors_relative_to_reference**
                                  2) + K1
            denominator = (inner_value**2) + 1.0
            omega_ast_p = -(numerator / denominator) * sigma_p
        return omega_ast, omega_ast_p


def BmatMRP(
        attitude_errors_relative_to_reference: torch.Tensor) -> torch.Tensor:
    sigma_squared = attitude_errors_relative_to_reference**2
    b = (1 - sigma_squared) * torch.eye(3, dtype=torch.float64) + torch.outer(
        attitude_errors_relative_to_reference,
        attitude_errors_relative_to_reference) * 2.0
    Tensor = torch.tensor(
        [[
            0, -attitude_errors_relative_to_reference[2],
            attitude_errors_relative_to_reference[1]
        ],
         [
             attitude_errors_relative_to_reference[2], 0,
             -attitude_errors_relative_to_reference[0]
         ],
         [
             -attitude_errors_relative_to_reference[1],
             attitude_errors_relative_to_reference[0], 0
         ]],
        dtype=torch.float64,
    )
    b = b + Tensor
    return b
