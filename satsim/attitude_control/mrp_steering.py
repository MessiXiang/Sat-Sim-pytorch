__all__ = ['MrpSteering', 'MrpSteeringDict']

from typing import TypedDict

import torch

from satsim.architecture import Module
from satsim.utils import Bmat


class MrpSteeringDict(TypedDict):
    pass


class MrpSteering(Module[MrpSteeringDict]):

    def __init__(
        self,
        *args,
        k1: float,
        k3: float,
        omega_max: float,
        ignore_outer_loop_feed_forward: bool,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        self._k1 = k1
        self._k3 = k3
        self._omega_max = omega_max
        self._ignore_outer_loop_feed_forward = ignore_outer_loop_feed_forward

    def forward(
        self,
        state_dict: MrpSteeringDict,
        *args,
        attitude_error_BR: torch.Tensor,
        **kwargs,
    ) -> tuple[MrpSteeringDict, tuple[torch.Tensor, torch.Tensor]]:

        sigma_cubed = attitude_error_BR**3
        inner_value = (torch.pi / 2) / self._omega_max * (
            self._k1 * attitude_error_BR + self._k3 * sigma_cubed)
        omega_BR_B = -torch.tanh(inner_value) * self._omega_max

        dot_omega_BR_B = torch.zeros_like(attitude_error_BR)
        if not self._ignore_outer_loop_feed_forward:
            b_matrix = Bmat(attitude_error_BR)
            sigma_p = 0.25 * torch.matmul(
                b_matrix,
                omega_BR_B.unsqueeze(-1),
            ).squeeze(-1)

            numerator = 3 * self._k3 * (attitude_error_BR**2) + self._k1
            denominator = (inner_value**2) + 1.0
            dot_omega_BR_B = -(numerator / denominator) * sigma_p

        return state_dict, (omega_BR_B, dot_omega_BR_B)
