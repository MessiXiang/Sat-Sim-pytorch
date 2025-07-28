__all__ = ['MrpSteering', 'MrpSteeringDict']

from typing import TypedDict, Tuple
from satsim.architecture import Module
from satsim.utils import Bmat
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
        k1: float,
        k3: float,
        omega_max: float,
        ignore_outer_loop_feed_forward: bool,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self._k1 = k1
        self._k3 = k3
        self._omega_max = omega_max
        self._ignore_outer_loop_feed_forward = ignore_outer_loop_feed_forward

    def forward(
        self,
        state_dict: MrpSteeringDict,
        *args,
        sigma_BR: torch.Tensor,
        **kwargs,
    ) -> tuple[MrpSteeringDict, Tuple[torch.Tensor, torch.Tensor]]:

        sigma_cubed = sigma_BR**3
        inner_value = (torch.pi / 2) / self._omega_max * (
            self._k1 * sigma_BR + self._k3 * sigma_cubed)
        omega_body_relative_to_reference_in_body_frame = -torch.atan(
            inner_value) / (torch.pi / 2) * self._omega_max

        angular_acceleration_relative_to_reference_in_body_frame = torch.zeros_like(
            sigma_BR)
        if not self._ignore_outer_loop_feed_forward:
            B = Bmat(sigma_BR)
            sigma_p = 0.25 * torch.matmul(
                B,
                omega_body_relative_to_reference_in_body_frame.unsqueeze(-1),
            ).squeeze(-1)

            numerator = 3 * self._k3 * (sigma_BR**2) + self._k1
            denominator = (inner_value**2) + 1.0
            angular_acceleration_relative_to_reference_in_body_frame = -(
                numerator / denominator) * sigma_p

        return state_dict, (
            omega_body_relative_to_reference_in_body_frame,
            angular_acceleration_relative_to_reference_in_body_frame,
        )
