__all__ = ["Inertial3D", "Inertial3DDict"]

from typing import TypedDict

import torch

from satsim.architecture import Module


class Inertial3DDict(TypedDict):
    pass


class Inertial3D(Module[Inertial3DDict]):

    def __init__(
        self,
        sigma_R0N: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Generate the reference attitude trajectory for a general 3D inertial pointing. A corrected body frame
        will align with the desired reference frame.
        """
        super().__init__(*args, **kwargs)

        sigma_R0N = torch.tensor(
            [0.0, 0.0, 0.0],
            dtype=torch.float32) if sigma_R0N is None else sigma_R0N

        self.register_buffer(
            "_sigma_R0N",
            sigma_R0N,
            persistent=False,
        )

    @property
    def sigma_R0N(self) -> torch.Tensor:
        return self.get_buffer('_sigma_R0N')

    def forward(
        self,
        state_dict: Inertial3DDict,
        *args,
        **kwargs,
    ) -> tuple[Inertial3DDict, tuple[torch.Tensor, torch.Tensor,
                                     torch.Tensor]]:
        """This is the main method that gets called every time the module is updated.
            Args:
                state_dict (Inertial3DDict | None): The state dictionary of the module.

        """

        attitude_RN = self.get_buffer("sigma_R0N")
        angular_velocity_RN_N = torch.zeros_like(attitude_RN)
        angular_acceleration_RN_N = torch.zeros_like(attitude_RN)

        return state_dict, (
            attitude_RN, angular_velocity_RN_N, angular_acceleration_RN_N
        )  #return Inertial3DDict(), (sigma_RN, omega_RN_N, dot_omega_RN_N)
        #return Inertial3DDict(), (...)
