__all__ = ['GravityBody', 'PointMassGravityBody']

from abc import ABC, abstractmethod
from typing import Self

import torch
from torch import Tensor

from satsim.architecture import Module, VoidStateDict, constants


class GravityBody(ABC):

    def __init__(
        self,
        *args,
        name: str,
        gm: float,
        equatorial_radius: float,
        polar_radius: float | None = None,
        is_central: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._name = name
        if polar_radius is None:
            polar_radius = equatorial_radius

        self._gm = gm
        self._equatorial_radius = equatorial_radius
        self._polar_radius = polar_radius
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
    def compute_gravitational_acceleration(
        self,
        relative_position: Tensor,
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

    @classmethod
    def create_sun(cls, **kwargs) -> Self:
        return cls(
            name='SUN',
            gm=constants.MU_SUN * 1e9,
            equatorial_radius=constants.REQ_SUN,
            **kwargs,
        )

    @classmethod
    def create_earth(cls, **kwargs) -> Self:
        return cls(
            name='EARTH',
            gm=constants.MU_EARTH * 1e9,
            equatorial_radius=constants.REQ_EARTH,
            **kwargs,
        )


class PointMassGravityBody(GravityBody):

    def compute_gravitational_acceleration(
        self,
        relative_position: Tensor,
    ) -> tuple[VoidStateDict, tuple[Tensor]]:
        """
        Computes the gravitational field for a set of point masses at specified positions.

        Args:
            relative_position (Tensor): Position tensor with shape [
            num_positions, 3],
            representing the 3D position vectors of points relative to each planet.

        Returns:
            Tensor: Gravitational field tensor with shape [num_positions, 3],
                representing the total gravitational field at each position due to all planets.

        Notes:
            - The gravitational field is computed using the inverse-square law, where the force
            magnitude is proportional to -mu / r^3, and r is the distance to each planet.
            - The field contributions from all planets are summed along the planet dimension.
        """

        r = torch.norm(relative_position, dim=-1, keepdim=True)

        force_magnitude = -self._gm / (r**3)  # [b, num_position, 1]
        grav_field = force_magnitude * relative_position
        if grav_field.dim() > 1:
            grav_field = grav_field.sum(-2)
        return grav_field
