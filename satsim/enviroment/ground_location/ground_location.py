__all__ = [
    'GroundLocation',
    'GroundLocationStateDict',
    'AccessState',
]

from typing import NamedTuple, TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import TypedDict

from satsim.architecture import Module
from satsim.simulation.gravity import Ephemeris
from satsim.utils import DCM_PCPF2SEZ, LLA2PCPF, PCPF2LLA


class AccessState(NamedTuple):
    has_access: Tensor  # [n_sc] or scalar  whether the spacecraft has access to the ground location
    slant_range: Tensor  # [n_sc] or scalar  slant range to the ground location
    elevation: Tensor  # [n_sc] or scalar  elevation angle to the ground location
    azimuth: Tensor  # [n_sc] or scalar  azimuth angle to the ground location
    range_dot: Tensor  # [n_sc] or scalar  range rate to the ground location
    azimuth_dot: Tensor  # [n_sc] or scalar  azimuth rate to the ground location
    elevation_dot: Tensor  # [n_sc] or scalar  elevation rate to the ground location
    position_BL_L: Tensor  # [n_sc, 3] or [3]  Spacecraft position relative to the groundLocation in the SEZ frame.
    velocity_BL_L: Tensor  # [n_sc, 3] or [3]  Spacecraft velocity relative to the groundLocation in the SEZ frame.


class GroundLocationStateDict(TypedDict):
    position_LP_P: torch.Tensor
    direction_cosine_matrix_LP: torch.Tensor


class GroundLocation(Module[GroundLocationStateDict]):

    def __init__(
        self,
        *args,
        minimum_elevation: Tensor,
        maximum_range: Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if maximum_range is None:
            maximum_range = torch.full(minimum_elevation.shape,
                                       -1.).to(minimum_elevation)

        self.register_buffer(
            '_minimum_elevation',
            minimum_elevation,
            persistent=False,
        )
        self.register_buffer(
            '_maximum_range',
            maximum_range,
            persistent=False,
        )

    @property
    def minimum_elevation(self) -> Tensor:
        return self.get_buffer('_minimum_elevation')

    @property
    def maximum_range(self) -> Tensor:
        return self.get_buffer('_maximum_range')

    def reset(self) -> GroundLocationStateDict:
        state_dict = GroundLocationStateDict(
            position_LP_P=torch.zeros(3),
            direction_cosine_matrix_LP=torch.eye(3),
        )
        return state_dict

    def forward(
        self,
        state_dict: GroundLocationStateDict,
        position_BN_N: Tensor,
        velocity_BN_N: Tensor,
        ephemeris: Ephemeris,
        *args,
        **kwargs,
    ) -> tuple[GroundLocationStateDict, tuple[AccessState, Tensor, Tensor]]:
        """
        Forward pass for ground location module.
        
        Args:
            state_dict: Current state dictionary
            spacecraft_states: Spacecraft state information
            planet_state: Planet state information
            position_LP_P: Position vector from planet center to ground location in planet-fixed frame
            direction_cosine_matrix_Planet2Location: Direction cosine matrix from planet-fixed to ground location frame
            
        Returns:
            Updated state dictionary with access information
        """
        # B for body or spacecraft
        # N for inertial frame
        # P for planet-fixed frame
        # L for location frame
        location_position_in_planet = state_dict['position_LP_P']
        direction_cosine_matrix_LP = state_dict['direction_cosine_matrix_LP']
        position_LP_N, position_LN_N, angular_velocity_PN_N, position_LP_N_unit = self.update_inertial_positions(
            location_position_in_planet,
            ephemeris,
        )

        # Expand single spacecraft to 2 dimension
        if position_BN_N.dim() == 1:
            position_BN_N = position_BN_N.unsqueeze(0)  # [1, 3]
            velocity_BN_N = velocity_BN_N.unsqueeze(0)  # [1, 3]

        # Expand ground/planet state
        position_PN_N = ephemeris['position_CN_N']
        direction_cosine_matrix_PN = ephemeris['direction_cosine_matrix_CN']

        position_BP_N = position_BN_N - position_PN_N
        position_BL_N = position_BP_N - position_LP_N
        position_BL_N_unit = F.normalize(position_BL_N, dim=-1)

        # Calculate elevation angle
        dot_products = torch.einsum(
            '...i, ...i -> ...',
            position_LP_N_unit,
            position_BL_N_unit,
        ).clamp(-1.0, 1.0)
        view_angle = torch.asin(
            dot_products
        )  # [n_sc] - elevation angle is angle between line of sight and horizontal

        # Transform to ground station coordinates

        direction_cosine_matrix_LN = torch.einsum(
            '...ij, ...jk -> ...ik',
            direction_cosine_matrix_LP,
            direction_cosine_matrix_PN,
        )  # [3, 3]
        position_BL_L = torch.einsum(
            '...ij, ...j -> ...i',
            direction_cosine_matrix_LN,
            position_BL_N,
        )

        # Calculate azimuth
        xy = position_BL_L[..., :2]
        xy_norm = torch.norm(xy, dim=-1)
        cos_azimuth = -position_BL_L[..., 0:1] / xy_norm
        sin_azimuth = position_BL_L[..., 1:2] / xy_norm
        azimuth = torch.atan2(sin_azimuth, cos_azimuth).squeeze(-1)

        # Velocity in local frame

        omega_PN_cross_position_BP_N = torch.cross(
            angular_velocity_PN_N.unsqueeze(-2),
            position_BP_N,
            dim=-1,
        )
        velocity_BL_N = velocity_BN_N - omega_PN_cross_position_BP_N  # [n_sc,3]
        velocity_BL_L = torch.einsum(
            '...ij, ...j -> ...i',
            direction_cosine_matrix_LN,
            velocity_BL_N,
        )

        # Range rate
        position_BL_N_norm = torch.norm(position_BL_N, dim=-1, keepdim=True)
        range_dot = torch.sum(velocity_BL_L * position_BL_L,
                              dim=-1) / position_BL_N_norm

        # Azimuth rate
        azimuth_dot = (-position_BL_L[..., 0] * velocity_BL_L[..., 1] +
                       position_BL_L[..., 1] * velocity_BL_L[..., 0]) / (
                           xy_norm**2)

        # Elevation rate
        elevation_dot = (velocity_BL_L[..., 2] / xy_norm -
                         position_BL_L[..., 2] *
                         (position_BL_L[..., 0] * velocity_BL_L[..., 0] +
                          position_BL_L[..., 1] * velocity_BL_L[..., 1]) /
                         (xy_norm**3)) / (1 +
                                          (position_BL_L[..., 2] / xy_norm)**2)

        # Determine visibility
        access_mask = (view_angle > self.minimum_elevation) & (
            (position_BL_N_norm.squeeze(-1) <= self.maximum_range) |
            (self.maximum_range < 0))

        access_state = AccessState(
            has_access=access_mask.to(dtype=torch.uint8),
            slant_range=position_BL_N_norm.squeeze(-1),
            elevation=view_angle,
            azimuth=azimuth,
            range_dot=range_dot,
            azimuth_dot=azimuth_dot,
            elevation_dot=elevation_dot,
            position_BL_L=position_BL_L,
            velocity_BL_L=velocity_BL_L,
        )

        return state_dict, (access_state, position_LP_N, position_LN_N)

    def specify_location_LLA(
        self,
        state_dict: GroundLocationStateDict,
        latitude: Tensor,
        longitude: Tensor,
        altitude: Tensor,
        planet_radius: float,
        polar_radius: float,
    ) -> GroundLocationStateDict:
        """
        Specify the ground location from planet-centered latitude, longitude, altitude position.
        
        Args:
            latitude: Latitude in radians
            longitude: Longitude in radians  
            altitude: Altitude in meters
            
        Returns:
            Tuple of (position_LP_P, direction_cosine_matrix_Planet2Inertial)
        """
        position_LP_P = LLA2PCPF(
            latitude,
            longitude,
            altitude,
            planet_radius,
            polar_radius,
        )
        direction_cosine_matrix_LP = DCM_PCPF2SEZ(latitude, longitude)

        state_dict['position_LP_P'] = position_LP_P
        state_dict['direction_cosine_matrix_LP'] = direction_cosine_matrix_LP

        return state_dict

    def specify_location_PCPF(
        self,
        state_dict: GroundLocationStateDict,
        position_LP_P: Tensor,
        planet_radius: float,
        polar_radius: float,
    ) -> GroundLocationStateDict:
        """
        Specify the ground location from planet-centered, planet-fixed coordinates.
        
        Args:
            position_LP_P: Position in planet-centered, planet-fixed coordinates (Tensor)
            
        Returns:
            Tuple of (position_LP_P, direction_cosine_matrix_Planet2Inertial)
        """
        tmp_llaposition = PCPF2LLA(
            position_LP_P,
            planet_radius,
            polar_radius,
        )
        direction_cosine_matrix_LP = DCM_PCPF2SEZ(tmp_llaposition[..., 0],
                                                  tmp_llaposition[..., 1])

        state_dict['position_LP_P'] = position_LP_P
        state_dict['direction_cosine_matrix_LP'] = direction_cosine_matrix_LP

        return state_dict

    def update_inertial_positions(
        self,
        position_LP_P: Tensor,
        planet_state: Ephemeris,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Update inertial positions and compute angular velocity.
        
        Args:
            state_dict: Current state dictionary
            position_LP_P: Position vector from planet center to ground location in planet-fixed frame
            planet_state: Planet state information
            
        Returns:
            Tuple of (position_LP_P, position_LN_N, omega_Planet [3], position_LP_N_unit [3])
        """
        direction_cosine_matrix_CN = planet_state['direction_cosine_matrix_CN']
        direction_cosine_matrix_CN_dot = planet_state[
            'direction_cosine_matrix_CN_dot']
        position_PN_N = planet_state['position_CN_N']

        # Transform position to inertial frame
        position_LP_N = torch.matmul(
            direction_cosine_matrix_CN.transpose(-1, -2),
            position_LP_P.unsqueeze(-1),
        ).squeeze(-1)
        position_LP_N_unit = position_LP_N / torch.norm(position_LP_N)
        position_LN_N = position_PN_N + position_LP_N

        # Compute angular velocity
        angular_velocity_tilde_PN_N = -torch.einsum(
            '...ij, ...kj -> ...ik',
            direction_cosine_matrix_CN_dot,
            direction_cosine_matrix_CN,
        )
        angular_velocity_PN_N = torch.stack([
            angular_velocity_tilde_PN_N[2, 1],
            angular_velocity_tilde_PN_N[0, 2], angular_velocity_tilde_PN_N[1,
                                                                           0]
        ])

        return position_LP_N, position_LN_N, angular_velocity_PN_N, position_LP_N_unit
