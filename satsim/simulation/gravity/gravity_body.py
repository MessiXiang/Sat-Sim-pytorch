__all__ = ['GravityBody', 'PointMassGravityBody']

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from satsim.architecture import Module, VoidStateDict


class GravityBody(Module[VoidStateDict], ABC):

    def __init__(
        self,
        *args,
        name: str,
        gm: float,
        equatorial_radius: float,
        polar_radius: float,
        is_central: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._name = name
        self.register_buffer(
            '_gm',
            torch.tensor([gm]),
            persistent=False,
        )
        self.register_buffer(
            '_equatorial_radius',
            torch.tensor([equatorial_radius]),
            persistent=False,
        )
        self.register_buffer(
            '_polar_radius',
            torch.tensor([polar_radius]),
            persistent=False,
        )

        self._is_central = is_central

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_central(self) -> bool:
        return self._is_central

    def set_central(self):
        self._is_central = True

    @abstractmethod
    def forward(
        self,
        relative_position: Tensor,  # [b, num_position, 3]
        *args,
        **kwargs,
    ) -> tuple[VoidStateDict, tuple[Tensor]]:
        """
        Computes the gravitational field for a set of point masses at specified positions.
        
        Args:
            position (Tensor): Position tensor with shape [batch_size, num_positions, 3],
                representing the 3D position vectors of points relative to each planet.
        
        Returns:
            Tensor: Gravitational field tensor with shape [batch_size, num_positions, 3],
                representing the total gravitational field at each position due to all planets.
        """
        pass


class PointMassGravityBody(GravityBody):

    def forward(
        self,
        relative_position: Tensor,  # [b, num_position, 3]
        state_dict: VoidStateDict | None = None,
        *args,
        **kwargs,
    ) -> tuple[VoidStateDict, tuple[Tensor]]:
        """
        Computes the gravitational field for a set of point masses at specified positions.

        Args:
            relative_position (Tensor): Position tensor with shape [batch_size, num_positions, num_planets, 3],
                representing the 3D position vectors of points relative to each planet.

        Returns:
            Tensor: Gravitational field tensor with shape [batch_size, num_positions, 3],
                representing the total gravitational field at each position due to all planets.

        Notes:
            - The gravitational field is computed using the inverse-square law, where the force
            magnitude is proportional to -mu / r^3, and r is the distance to each planet.
            - The field contributions from all planets are summed along the planet dimension.
        """

        r = torch.norm(relative_position, dim=-1, keepdim=True)
        gm = self.get_buffer('_gm')
        force_magnitude = -gm / (r**3)  # [b, num_position, 1]
        grav_field = force_magnitude * relative_position
        if grav_field.dim() > 1:
            grav_field = grav_field.sum(-2)
        return dict(), (grav_field, )
