__all__ = [
    'GravityEffector',
    'Ephemeris',
    'GravityEffectorStateDict',
]

from typing import Optional, NotRequired
import torch
from torch import Tensor
from typing_extensions import TypedDict
from satsim.architecture import Module


class Ephemeris(TypedDict):
    """
    All Tensor should be of shape [batch_size, num_body, *value_shape]. First two dimension can be ommited or set to 1
    
    """
    position_in_inertial: Tensor  # position vector
    velocity_in_inertial: Tensor  # velocity vector
    J2000_2_planet_fixed: Tensor  # DCM from J2000 to planet-fixed
    J2000_2_planet_fixed_dot: Tensor  # DCM derivative


class GravityEffectorStateDict(TypedDict):
    planet_bodies_ephemeris: Ephemeris | None  # list of Ephemeris of planet bodies (non central body)
    central_body_ephemeris: Ephemeris | None  # Ephemeris of central body


class GravityEffector(Module[GravityEffectorStateDict]):

    def __init__(
        self,
        *args,
        central_body_mu: float,
        planet_bodies_mu: list[float] | None = None,
        planet_bodies_ephemeris_init: Ephemeris
        | None = None,
        central_body_ephemeris_init: Ephemeris
        | None = None,
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

        self.planet_bodies_init = planet_bodies_ephemeris_init
        self.central_body_init = central_body_ephemeris_init

    @property
    def num_planet_bodies(self):
        planet_bodies_mu = self.get_buffer('planet_bodies_mu')
        return planet_bodies_mu.size(-1)

    @property
    def all_grav_body_mu(self):
        planet_bodies_mu = self.get_buffer('planet_bodies_mu')
        central_body_mu = self.get_buffer('central_body_mu')
        return torch.cat([planet_bodies_mu, central_body_mu], dim=-2)

    def reset(self) -> GravityEffectorStateDict:
        state_dict = super().reset()
        state_dict.update({
            'planet_bodies': self.planet_bodies_init,
            'central_body': self.central_body_init,
        })
        return state_dict

    def compute_gravity_field(
        self,
        state_dict: GravityEffectorStateDict,
        position_spacecraft_wrt_central_body_in_inertial:
        Tensor,  # [b, num_spacecraft, 3]
        central_body_ephemeris: Ephemeris,
        planet_body_ephemeris: Ephemeris | None = None,
    ) -> Tensor:
        position_central_body_in_inertial = self._calculate_next_position(
            central_body_ephemeris)  # [b, 1, 3]
        position_spacecraft_in_inertial = position_spacecraft_wrt_central_body_in_inertial + position_central_body_in_inertial

        acceleration_spacecraft_wrt_central_body_in_inertial = torch.zeros_like(
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
        acceleration_spacecraft_wrt_central_body_in_inertial = acceleration_spacecraft_wrt_central_body_in_inertial + self.compute_point_mass_field(
            self.get_buffer('planet_bodies_mu'),  # [b, num_planet, 1]
            (position_planet_in_inertial -
             position_central_body_in_inertial).unsqueeze(-4),
        )  # [b, 1, 3]

        acceleration_spacecraft_wrt_central_body_in_inertial = acceleration_spacecraft_wrt_central_body_in_inertial + self.compute_point_mass_field(
            self.all_grav_body_mu,
            position_spacecraft_wrt_planet_in_inerital,
        )

        return acceleration_spacecraft_wrt_central_body_in_inertial

    def update_inertial_position_and_velocity(
        self,
        state_dict: GravityEffectorStateDict,
        r_spacecraft2frame_inertial: Tensor,
        rDot_spacecraft2frame_inertial: Tensor,
    ) -> tuple[Tensor, Tensor]:

        inertial_position_property = torch.zeros_like(
            r_spacecraft2frame_inertial)
        inertial_velocity_property = torch.zeros_like(
            rDot_spacecraft2frame_inertial)

        if state_dict['central_body'] is not None:
            r_center2inertial_inertial = self._calculate_next_position(
                state_dict['central_body'])
            inertial_position_property = r_center2inertial_inertial + r_spacecraft2frame_inertial
            inertial_velocity_property = state_dict['central_body'][
                'velocity_in_inertial'] + rDot_spacecraft2frame_inertial
        else:
            inertial_position_property = r_spacecraft2frame_inertial
            inertial_velocity_property = rDot_spacecraft2frame_inertial

        return inertial_position_property, inertial_velocity_property

    def _calculate_next_position(
        self,
        ephemeris: Ephemeris,
    ) -> Tensor:
        """Euler integration to compute planet position"""
        dt = self._timer.dt
        return ephemeris[
            'position_in_inertial'] + ephemeris['velocity_in_inertial'] * dt

    def update_energy_contributions(
            self, state_dict: GravityEffectorStateDict,
            r_spacecraft2frame_inertial: Tensor) -> float:

        orbit_potential_energy_contribution = torch.tensor(
            0.0, device=r_spacecraft2frame_inertial.device)

        if state_dict['central_body'] is not None:
            r_center2inertial_inertial = self._calculate_next_position(
                state_dict['central_body'])
            r_spacecraft2inertial_inertial = r_spacecraft2frame_inertial + r_center2inertial_inertial

            orbit_potential_energy_contribution = orbit_potential_energy_contribution + self.compute_point_mass_potential(
                self.planet_bodies_mu[-1], r_spacecraft2frame_inertial)
        else:
            r_spacecraft2inertial_inertial = r_spacecraft2frame_inertial

        for index, planet_body in enumerate(state_dict['planet_bodies']):
            r_planet2inertial_inertial = self._calculate_next_position(
                planet_body)
            # relative dynamics correction
            if state_dict['central_body'] is not None:
                r_planet2center_inertial = r_planet2inertial_inertial - r_center2inertial_inertial
                orbit_potential_energy_contribution = orbit_potential_energy_contribution + self.compute_point_mass_potential(
                    self.planet_bodies_mu[index], r_planet2center_inertial)
            # potential energy in the current planet field
            r_spacecraft2planet_inertial = r_spacecraft2inertial_inertial - r_planet2inertial_inertial
            orbit_potential_energy_contribution = orbit_potential_energy_contribution + self.compute_point_mass_potential(
                self.planet_bodies_mu[index], r_spacecraft2planet_inertial)

        return orbit_potential_energy_contribution

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

        grav_acceleration_in_inertial = self.compute_point_mass_field(
            mu_body, position_wrt_planet_in_inertial)

        # convert back to inertial frame
        return grav_acceleration_in_inertial

    def compute_point_mass_field(
            self,
            mu: Tensor,  # [b, num_planet, 1]
            position: Tensor,  # [b, num_position, num_planet, 3]
    ) -> Tensor:
        r = torch.norm(position, dim=-1, keepdim=True)
        force_magnitude = -mu.unsqueeze(-3) / (
            r**3)  # [b, num_position, num_planet, 1]
        grav_field = force_magnitude * position
        return grav_field.sum(-2)

    def compute_point_mass_potential(self, mu: float,
                                     position: Tensor) -> Tensor:
        r = torch.norm(position)
        if r < 1e-6:
            return torch.tensor(0.0,
                                dtype=position.dtype,
                                device=position.device)
        return -mu / r
