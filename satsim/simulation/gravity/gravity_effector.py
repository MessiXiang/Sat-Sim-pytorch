__all__ = [
    'GravityField',
]
from typing import Iterable

import torch
from torch import Tensor

from satsim.architecture import Module, VoidStateDict
from satsim.utils import move_to

from .gravity_body import GravityBody
from .spice_interface import (Ephemeris, SpiceInterface, string_normalizer,
                              zero_ephemeris)


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

        self._gravity_bodies_ephemeris: Ephemeris = move_to(
            gravity_bodies_ephemeris,
            target=target,
        )

    def _calculate_next_position(self, ) -> None:
        """Euler integration to compute planet position"""
        dt = self._timer.dt

        ephemeris = self._gravity_bodies_ephemeris
        self._position_CN_N = ephemeris[
            'position_CN_N'] + ephemeris['velocity_CN_N'] * dt

    def forward(
        self,
        position_BPc_N: Tensor,  #[b, ns, 3]
        state_dict: VoidStateDict | None = None,
        *args,
        **kwargs,
    ) -> tuple[VoidStateDict, tuple[Tensor]]:
        """position_BPc_N(Tensor)  [b, num_spacecraft, 3]"""
        self._load_ephemeris(position_BPc_N)
        self._calculate_next_position()

        if self.central_gravity_body is not None:
            position_BN_N = \
                position_BPc_N + \
                self._position_CN_N[..., self._central_gravity_body_idx, :]
        else:
            position_BN_N = \
                position_BPc_N

        position_CN_N = self._position_CN_N.unsqueeze(-2)
        position_BC_N = \
            (position_BN_N - \
        position_CN_N).transpose(-2,-3)  # [b, num_spacecraft, num_planet, 3]

        # Subtract acceleration of central body due to other bodies to
        # get relative acceleration of spacecraft. See Vallado on
        # "Three-body and n-body Equations"
        accelerations_PcC_N = []  # [b,(1),3]
        accelerations_BC_N = []  #[b,ns,3]
        for idx, gravity_body_name in enumerate(self._gravity_bodies_names):
            gravity_body: GravityBody = self.get_submodule(
                f'_gravity_body_{gravity_body_name}')
            if (self._central_body_name is not None
                    and self._central_body_name != gravity_body_name):
                position_CPc_N = \
                    self._position_CN_N[...,idx,:] \
                    - self._position_CN_N[...,self._central_gravity_body_idx,:]

                acceleration_PcC_N = gravity_body.compute_gravitational_acceleration(
                    relative_position=position_CPc_N)

                accelerations_PcC_N.append(acceleration_PcC_N)

            position_BC_N = \
                position_BC_N[
                ..., idx, :]
            acceleration_BC_N = gravity_body.compute_gravitational_acceleration(
                relative_position=position_BC_N)
            accelerations_BC_N.append(acceleration_BC_N)

        if len(accelerations_PcC_N) > 0:
            acceleration_PcN_N = torch.sum(
                torch.stack(accelerations_PcC_N),
                dim=0,
            )
        else:
            acceleration_PcN_N = 0.

        acceleration_BN_N = torch.sum(torch.stack(accelerations_BC_N), dim=0)

        return dict(), (acceleration_BN_N + acceleration_PcN_N, )

    def update_inertial_position_and_velocity(
        self,
        position_BPc_N: Tensor,
        velocity_BPc_N: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Update the inertial position and velocity of the spacecraft based on the current ephemeris of the central body.
        Args:
            position_BPc_N (Tensor): Position of the spacecraft relative to the central body in inertial frame.
                Shape: [batch_size, num_spacecraft, 3]
            velocity_BPc_N (Tensor): Velocity of the spacecraft relative to the central body in inertial frame.
                Shape: [batch_size, num_spacecraft, 3]
            central_body_ephemeris (Ephemeris): Ephemeris data for the central body, containing position and velocity in inertial frame.
        Returns:
            tuple[Tensor, Tensor]: Updated position and velocity of the spacecraft in inertial frame.
                Shapes: ([batch_size, num_spacecraft, 3], [batch_size, num_spacecraft, 3])
        """
        position_PcN_N = \
            self._position_CN_N[
            ..., self._central_gravity_body_idx, :]  # [b, 1, 3]

        position_BN_N = (position_BPc_N + position_PcN_N)

        velocity_BN_N = (velocity_BPc_N +
                         self._gravity_bodies_ephemeris['velocity_CN_N'][
                             ..., self._central_gravity_body_idx, :])

        return (
            position_BN_N,
            velocity_BN_N,
        )
