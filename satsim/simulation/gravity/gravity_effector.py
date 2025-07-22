__all__ = [
    'GravityField',
    'PointMassGravityField',
    'Ephemeris',
]
from abc import ABC, abstractmethod
from typing import Self
import torch
from torch import Tensor
from typing_extensions import TypedDict

from satsim.architecture import Module, VoidStateDict


class Ephemeris(TypedDict):
    """
    All Tensor should be of shape [batch_size, num_body, *value_shape]. First two dimension can be ommited or set to 1
    
    """
    position_in_inertial: Tensor  # position vector
    velocity_in_inertial: Tensor  # velocity vector
    J2000_2_planet_fixed: Tensor  # DCM from J2000 to planet-fixed
    J2000_2_planet_fixed_dot: Tensor  # DCM derivative


class GravityField(Module[VoidStateDict], ABC):

    def __init__(
        self,
        *args,
        central_body_mu: float,
        planet_bodies_mu: list[float] | None = None,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        if planet_bodies_mu is None:
            planet_bodies_mu = []

        self.register_buffer(
            'planet_bodies_mu',
            torch.tensor(planet_bodies_mu),
            persistent=False,
        )
        self.register_buffer(
            'central_body_mu',
            torch.tensor(central_body_mu),
            persistent=False,
        )

    @property
    def num_planet_bodies(self):
        planet_bodies_mu = self.get_buffer('planet_bodies_mu')
        return planet_bodies_mu.size(-1)

    @property
    def all_grav_body_mu(self):
        planet_bodies_mu = self.get_buffer('planet_bodies_mu')
        central_body_mu = self.get_buffer('central_body_mu')
        return torch.cat([planet_bodies_mu, central_body_mu], dim=-2)

    def compute_gravity_field(
        self,
        position_spacecraft_wrt_central_body_in_inertial:
        Tensor,  # [b, num_spacecraft, 3]
        central_body_ephemeris: Ephemeris,
        planet_body_ephemeris: Ephemeris | None = None,
    ) -> Tensor:
        position_central_body_in_inertial = self._calculate_next_position(
            central_body_ephemeris)  # [b, 1, 3]
        position_spacecraft_in_inertial = position_spacecraft_wrt_central_body_in_inertial + position_central_body_in_inertial

        acceleration_spacecraft_in_inertial = torch.zeros_like(
            position_spacecraft_wrt_central_body_in_inertial)

        position_planet_in_inertial = self._calculate_next_position(
            planet_body_ephemeris)  # [b, num_planet,3]
        all_body_next_position_in_inertial = torch.cat(
            [position_planet_in_inertial, position_central_body_in_inertial],
            dim=-2,
        )
        position_spacecraft_wrt_planet_in_inerital = position_spacecraft_in_inertial.unsqueeze(
            -2) - all_body_next_position_in_inertial.unsqueeze(
                -3)  # [b, num_spacecraft, num_planet + 1, 3]

        # Subtract acceleration of central body due to other bodies to
        # get relative acceleration of spacecraft. See Ballado on
        # "Three-body and n-body Equations"
        acceleration_spacecraft_in_inertial = acceleration_spacecraft_in_inertial + self.compute_field(
            mu=self.get_buffer('planet_bodies_mu'),  # [b, num_planet, 1]
            position=(position_planet_in_inertial -
                      position_central_body_in_inertial).unsqueeze(-3),
        )  # [b, 1, 3]

        acceleration_spacecraft_in_inertial = acceleration_spacecraft_in_inertial + self.compute_field(
            mu=self.all_grav_body_mu,
            position=position_spacecraft_wrt_planet_in_inerital,
        )

        return acceleration_spacecraft_in_inertial

    def update_inertial_position_and_velocity(
        self,
        position_spacecraft_wrt_central_body_in_inertial: Tensor,
        velocity_spacecraft_wrt_central_body_in_inertial: Tensor,
        central_body_ephemeris: Ephemeris,
    ) -> tuple[Tensor, Tensor]:
        """
        Update the inertial position and velocity of the spacecraft based on the current ephemeris of the central body.
        Args:
            position_spacecraft_wrt_central_body_in_inertial (Tensor): Position of the spacecraft relative to the central body in inertial frame.
                Shape: [batch_size, num_spacecraft, 3]
            velocity_spacecraft_wrt_central_body_in_inertial (Tensor): Velocity of the spacecraft relative to the central body in inertial frame.
                Shape: [batch_size, num_spacecraft, 3]
            central_body_ephemeris (Ephemeris): Ephemeris data for the central body, containing position and velocity in inertial frame.
        Returns:
            tuple[Tensor, Tensor]: Updated position and velocity of the spacecraft in inertial frame.
                Shapes: ([batch_size, num_spacecraft, 3], [batch_size, num_spacecraft, 3])
        """
        position_central_body_in_inertial = self._calculate_next_position(
            central_body_ephemeris)  # [b, 1, 3]

        position_spacecraft_in_inertial = (
            position_spacecraft_wrt_central_body_in_inertial +
            position_central_body_in_inertial)

        velocity_spacecraft_in_inertial = (
            velocity_spacecraft_wrt_central_body_in_inertial +
            central_body_ephemeris['velocity_in_inertial'])

        return (
            position_spacecraft_in_inertial,
            velocity_spacecraft_in_inertial,
        )

    def _calculate_next_position(
        self,
        ephemeris: Ephemeris,
    ) -> Tensor:
        """Euler integration to compute planet position"""
        dt = self._timer.dt
        return ephemeris[
            'position_in_inertial'] + ephemeris['velocity_in_inertial'] * dt

    def compute_gravity_inertial(
        self,
        body: Ephemeris,
        mu_body: float,
        position_wrt_planet_in_inertial: Tensor,
    ) -> Tensor:
        direction_cos_matrix_inertial_2_planet_fixed = body[
            'J2000_2_planet_fixed']

        direction_cos_matrix_inertial_2_planet_fixed_dot = body[
            'J2000_2_planet_fixed_dot']
        direction_cos_matrix_inertial_2_planet_fixed = direction_cos_matrix_inertial_2_planet_fixed + direction_cos_matrix_inertial_2_planet_fixed_dot * self._timer.dt

        body[
            'J2000_2_planet_fixed'] = direction_cos_matrix_inertial_2_planet_fixed
        body[
            'J2000_2_planet_fixed_dot'] = direction_cos_matrix_inertial_2_planet_fixed_dot

        # compute position in planet-fixed frame

        grav_acceleration_in_inertial = self.compute_field(
            mu_body, position_wrt_planet_in_inertial)

        # convert back to inertial frame
        return grav_acceleration_in_inertial

    @abstractmethod
    def compute_field(
        self,
        mu: Tensor,  # [b, num_planet, 1]
        position: Tensor,  # [b, num_position, num_planet, 3]
        *args,
        **kwargs,
    ) -> Tensor:
        """
        Computes the gravitational field for a set of point masses at specified positions.
        
        Args:
            mu (Tensor): Gravitational parameter tensor with shape [batch_size, num_planets, 1],
                representing the gravitational constant times the mass of each planet.
            position (Tensor): Position tensor with shape [batch_size, num_positions, num_planets, 3],
                representing the 3D position vectors of points relative to each planet.
        
        Returns:
            Tensor: Gravitational field tensor with shape [batch_size, num_positions, 3],
                representing the total gravitational field at each position due to all planets.
        """
        pass


class PointMassGravityField(GravityField):

    def compute_field(
        self,
        mu: Tensor,  # [b, num_planet, 1]
        position: Tensor,  # [b, num_position, num_planet, 3]
        *args,
        **kwargs,
    ) -> Tensor:
        """
        Computes the gravitational field for a set of point masses at specified positions.

        Args:
            mu (Tensor): Gravitational parameter tensor with shape [batch_size, num_planets, 1],
                representing the gravitational constant times the mass of each planet.
            position (Tensor): Position tensor with shape [batch_size, num_positions, num_planets, 3],
                representing the 3D position vectors of points relative to each planet.

        Returns:
            Tensor: Gravitational field tensor with shape [batch_size, num_positions, 3],
                representing the total gravitational field at each position due to all planets.

        Notes:
            - The gravitational field is computed using the inverse-square law, where the force
            magnitude is proportional to -mu / r^3, and r is the distance to each planet.
            - The field contributions from all planets are summed along the planet dimension.
        """
        r = torch.norm(position, dim=-1, keepdim=True)
        force_magnitude = -mu.unsqueeze(-3) / (
            r**3)  # [b, num_position, num_planet, 1]
        grav_field = force_magnitude * position
        return grav_field.sum(-2)
