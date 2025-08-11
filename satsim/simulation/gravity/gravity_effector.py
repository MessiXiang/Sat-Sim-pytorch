__all__ = [
    'GravityField',
]
from typing import Iterable

import torch
from torch import Tensor

from satsim.architecture import Module, VoidStateDict
from satsim.utils import move_to

from .gravity_body import GravityBody
from .spice_interface import SpiceInterface, string_normalizer, zero_ephemeris


class GravityField(Module[VoidStateDict]):

    def __init__(
        self,
        *args,
        spice_interface: SpiceInterface | None = None,
        gravity_bodies: Iterable[GravityBody] | GravityBody,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._spice_interface = spice_interface

        if isinstance(gravity_bodies, GravityBody):
            gravity_bodies = [gravity_bodies]

        gravity_bodies_names: list[str] = []
        self._central_body_name: str | None = None
        self._central_gravity_body_idx: str | None = None
        for idx, gravity_body in enumerate(gravity_bodies):
            gravity_body_name = string_normalizer(gravity_body.name)
            if gravity_body_name in gravity_bodies_names:
                raise ValueError('Multiple body have same name')

            if gravity_body.is_central:
                if self._central_body_name is not None:
                    raise ValueError('Multiple central body')
                self._central_body_name = gravity_body_name
                self._central_gravity_body_idx = idx

            gravity_bodies_names.append(gravity_body_name)
            self.add_module(f'_gravity_body_{gravity_body_name}', gravity_body)

        self._gravity_bodies_names = gravity_bodies_names

    def get_gravity_body(self, gravity_body_name: str):
        return self.get_submodule(f'_gravity_body_{gravity_body_name}')

    @property
    def gravity_bodies_names(self) -> list[str]:
        return self._gravity_bodies_names

    @property
    def spice_interface(self) -> SpiceInterface | None:
        return self._spice_interface

    @property
    def central_gravity_body(self) -> GravityBody | None:
        if self._central_body_name is None:
            return None

        return self.get_submodule(f'_gravity_body_{self._central_body_name}')

    def _load_ephemeris(self, target: torch.Tensor) -> None:

        if self.spice_interface is None:
            gravity_bodies_ephemeris = zero_ephemeris()
        else:
            _, (gravity_bodies_ephemeris, ) = self.spice_interface(
                names=self.gravity_bodies_names, )

        self._gravity_bodies_ephemeris = move_to(
            gravity_bodies_ephemeris,
            target=target,
        )

    def _calculate_next_position(self, ) -> None:
        """Euler integration to compute planet position"""
        dt = self._timer.dt

        ephemeris = self._gravity_bodies_ephemeris
        self._gravity_bodies_position_in_inertial = ephemeris[
            'position_in_inertial'] + ephemeris['velocity_in_inertial'] * dt

    def forward(
        self,
        position_spacecraft_wrt_central_point_in_inertial: Tensor,  #[b, ns, 3]
        state_dict: VoidStateDict | None = None,
        *args,
        **kwargs,
    ) -> tuple[VoidStateDict, tuple[Tensor]]:
        """position_spacecraft_wrt_central_body_in_inertial(Tensor)  [b, num_spacecraft, 3]"""
        self._load_ephemeris(position_spacecraft_wrt_central_point_in_inertial)
        self._calculate_next_position()

        if self.central_gravity_body is not None:
            position_spacecraft_in_inertial = \
                position_spacecraft_wrt_central_point_in_inertial + \
                self._gravity_bodies_position_in_inertial[..., self._central_gravity_body_idx, :]
        else:
            position_spacecraft_in_inertial = \
                position_spacecraft_wrt_central_point_in_inertial

        position_spacecraft_in_inertial = position_spacecraft_in_inertial.unsqueeze(
            -2)
        position_spacecraft_wrt_planet_in_inerital = \
            position_spacecraft_in_inertial - \
        self._gravity_bodies_position_in_inertial.expand_as(
            position_spacecraft_in_inertial,
        )  # [b, num_spacecraft, num_planet, 3]

        # Subtract acceleration of central body due to other bodies to
        # get relative acceleration of spacecraft. See Vallado on
        # "Three-body and n-body Equations"
        accelerations_central_body = []  # [b,(1),3]
        accelerations_spacecraft = []  #[b,ns,3]
        for idx, gravity_body_name in enumerate(self._gravity_bodies_names):
            gravity_body: GravityBody = self.get_submodule(
                f'_gravity_body_{gravity_body_name}')
            if (self._central_body_name is not None
                    and self._central_body_name != gravity_body_name):
                relative_position_gravity_body = \
                    self._gravity_bodies_position_in_inertial[...,idx,:] \
                    - self._gravity_bodies_position_in_inertial[...,self._central_gravity_body_idx,:]

                _, (acceleration_central_body,
                    ) = gravity_body.compute_gravitational_acceleration(
                        relative_position=relative_position_gravity_body)

                accelerations_central_body.append(acceleration_central_body)

            position_spacecraft_wrt_gravity_body_in_inertial = \
                position_spacecraft_wrt_planet_in_inerital[
                ..., idx, :]
            _, (acceleration_spacecraft,
                ) = gravity_body.compute_gravitational_acceleration(
                    relative_position=
                    position_spacecraft_wrt_gravity_body_in_inertial)
            accelerations_spacecraft.append(acceleration_spacecraft)

        if len(accelerations_central_body) > 0:
            total_acceleration_central_body = torch.sum(
                torch.stack(accelerations_central_body), dim=0)
        else:
            total_acceleration_central_body = 0.

        total_acceleration_spacecraft = torch.sum(
            torch.stack(accelerations_spacecraft), dim=0)

        return dict(), (total_acceleration_spacecraft +
                        total_acceleration_central_body, )

    def update_inertial_position_and_velocity(
        self,
        position_spacecraft_wrt_central_body_in_inertial: Tensor,
        velocity_spacecraft_wrt_central_body_in_inertial: Tensor,
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
        position_central_body_in_inertial = \
            self._gravity_bodies_position_in_inertial[
            ..., self._central_gravity_body_idx, :]  # [b, 1, 3]

        position_spacecraft_in_inertial = (
            position_spacecraft_wrt_central_body_in_inertial +
            position_central_body_in_inertial)

        velocity_spacecraft_in_inertial = (
            velocity_spacecraft_wrt_central_body_in_inertial +
            self._gravity_bodies_ephemeris['velocity_in_inertial'][
                ..., self._central_gravity_body_idx, :])

        return (
            position_spacecraft_in_inertial,
            velocity_spacecraft_in_inertial,
        )
