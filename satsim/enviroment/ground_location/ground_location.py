__all__ = [
    'GroundLocation',
    'GroundLocationStateDict',
    'Ephemeris',
    'SpaceCraftStateDict',
    'AccessStateDict',
]

from typing import Tuple
import torch
from torch import Tensor
from typing_extensions import TypedDict
from satsim.architecture import Module
from satsim.simulation.gravity import Ephemeris


# 定义数据结构
class GroundStateDict(TypedDict):
    position_LN_N: Tensor  # [3]  position vector of the ground location in inertial frame
    position_LP_N: Tensor  # [3]  position vector of the ground location in planet-fixed frame


class SpaceCraftStateDict(TypedDict):
    position_BN_N: Tensor  # [n_sc, 3] or [3]  spacecraft position in inertial frame
    velocity_BN_N: Tensor  # [n_sc, 3] or [3]  spacecraft velocity in inertial frame


class AccessStateDict(TypedDict):
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
    ground_state: GroundStateDict  # state of the ground location
    access_states: AccessStateDict  # access states for the specified ground location


class GroundLocation(Module[GroundLocationStateDict]):

    def __init__(
        self,
        *args,
        minimum_elevation: float = 10.0 * torch.pi / 180.0,
        maximum_range: float = -1.0,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self.register_buffer('_minimum_elevation',
                             torch.tensor(minimum_elevation),
                             persistent=False)
        self.register_buffer('_maximum_range',
                             torch.tensor(maximum_range),
                             persistent=False)

    @property
    def minimum_elevation(self) -> Tensor:
        return self.get_buffer('_minimum_elevation')

    @property
    def maximum_range(self) -> Tensor:
        return self.get_buffer('_maximum_range')

    def forward(
        self,
        state_dict: GroundLocationStateDict,
        position_BN_N: Tensor,
        velocity_BN_N: Tensor,
        planet_radius: Tensor,
        polar_radius: Tensor,
        ephemeris: Ephemeris,
        position_LP_P: Tensor,
        direction_cosine_matrix_Planet2Location: Tensor,
        *args,
        **kwargs,
    ) -> GroundLocationStateDict:
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
        state_dict, omega_planet, position_LP_N_unit = self.update_inertial_positions(
            state_dict,
            position_LP_P,
            ephemeris,
        )

        # Expand single spacecraft to 2 dimension
        if position_BN_N.dim() == 1:
            position_BN_N = position_BN_N.unsqueeze(0)  # [1, 3]
            velocity_BN_N = velocity_BN_N.unsqueeze(0)  # [1, 3]

        # Expand ground/planet state
        position_PN_N = ephemeris['position_in_inertial']
        direction_cosine_matrix_Inertial2Planet = ephemeris[
            'J2000_2_planet_fixed']

        position_BP_N = position_BN_N - position_PN_N
        position_LP_N = state_dict['ground_state']['position_LP_N']
        position_BL_N = position_BP_N - position_LP_N
        position_BL_norm = torch.norm(position_BL_N, dim=-1)

        relative_heading_N_unit = position_BL_N / position_BL_norm

        # Calculate elevation angle
        dot_products = (position_LP_N_unit *
                        relative_heading_N_unit).sum(dim=-1).clamp(-1.0, 1.0)
        view_angle = torch.asin(
            dot_products
        )  # [n_sc] - elevation angle is angle between line of sight and horizontal

        # Transform to ground station coordinates
        direction_cosine_matrix_full = torch.matmul(
            direction_cosine_matrix_Planet2Location,
            direction_cosine_matrix_Inertial2Planet)  # [3, 3]
        position_BL_L = torch.matmul(position_BL_N,
                                     direction_cosine_matrix_full.T)

        # Calculate azimuth
        xy = position_BL_L[..., :2]
        xy_norm = torch.norm(xy, dim=-1)
        cos_azimuth = -position_BL_L[..., 0:1] / xy_norm
        sin_azimuth = position_BL_L[..., 1:2] / xy_norm
        azimuth = torch.atan2(sin_azimuth, cos_azimuth).squeeze(-1)

        # Velocity in local frame
        omega_PN_cross_position_BP_N = torch.cross(omega_planet.unsqueeze(-2),
                                                   position_BP_N,
                                                   dim=-1)
        velocity_BL_N = velocity_BN_N - omega_PN_cross_position_BP_N
        velocity_BL_L = torch.matmul(velocity_BL_N,
                                     direction_cosine_matrix_full.T)

        # Range rate
        range_dot = torch.sum(velocity_BL_L * position_BL_L,
                              dim=-1) / position_BL_norm

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
            (position_BL_norm.squeeze(-1) <= self.maximum_range) |
            (self.maximum_range < 0))

        access_state_msg = {
            'slant_range': position_BL_norm.squeeze(-1),
            'elevation': view_angle,
            'azimuth': azimuth,
            'position_BL_L': position_BL_L,
            'velocity_BL_L': velocity_BL_L,
            'range_dot': range_dot,
            'azimuth_dot': azimuth_dot,
            'elevation_dot': elevation_dot,
            'has_access': access_mask.to(dtype=torch.uint8)
        }

        state_dict['access_states'] = access_state_msg

        return state_dict

    def specify_location(
        self,
        latitude: Tensor,
        longitude: Tensor,
        altitude: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Specify the ground location from planet-centered latitude, longitude, altitude position.
        
        Args:
            latitude: Latitude in radians
            longitude: Longitude in radians  
            altitude: Altitude in meters
            
        Returns:
            Tuple of (position_LP_P, direction_cosine_matrix_Planet2Inertial)
        """
        lla_postion = torch.stack([latitude, longitude, altitude])
        position_LP_P = self.LLA2PCPF(lla_postion, self.planet_radius,
                                      self.get_buffer("polar_radius"))
        direction_cosine_matrix_Planet2Inertial = self.DCM_PCPF2SEZ(
            latitude, longitude)

        return position_LP_P, direction_cosine_matrix_Planet2Inertial

    def specify_location_PCPF(self,
                              position_LP_P: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Specify the ground location from planet-centered, planet-fixed coordinates.
        
        Args:
            position_LP_P: Position in planet-centered, planet-fixed coordinates (Tensor)
            
        Returns:
            Tuple of (position_LP_P, direction_cosine_matrix_Planet2Inertial)
        """
        tmp_llaposition = self.PCPF2LLA(position_LP_P, self.planet_radius,
                                        self.get_buffer("polar_radius"))
        direction_cosine_matrix_Planet2Inertial = self.DCM_PCPF2SEZ(
            tmp_llaposition[0], tmp_llaposition[1])

        return (position_LP_P, direction_cosine_matrix_Planet2Inertial)

    def update_inertial_positions(
        self,
        state_dict: GroundLocationStateDict,
        position_LP_P: Tensor,
        planet_state: Ephemeris,
    ) -> Tuple[GroundLocationStateDict, Tensor, Tensor]:
        """
        Update inertial positions and compute angular velocity.
        
        Args:
            state_dict: Current state dictionary
            position_LP_P: Position vector from planet center to ground location in planet-fixed frame
            planet_state: Planet state information
            
        Returns:
            Tuple of (updated_state_dict, omega_Planet [3], position_LP_N_unit [3])ss
        """
        direction_cosine_matrix_Inertial2Planet = planet_state[
            'J2000_2_planet_fixed']
        direction_cosine_matrix_Inertial2Planet_dot = planet_state[
            'J2000_2_planet_fixed_dot']
        position_PN_N = planet_state['position_in_inertial']

        # Transform position to inertial frame
        position_LP_N = torch.matmul(direction_cosine_matrix_Inertial2Planet.T,
                                     position_LP_P)
        position_LP_N_unit = position_LP_N / torch.norm(position_LP_N)
        position_LN_N = position_PN_N + position_LP_N

        # Compute angular velocity
        omega_tilde_PN = -torch.matmul(
            direction_cosine_matrix_Inertial2Planet_dot,
            direction_cosine_matrix_Inertial2Planet.T)
        omega_PN = torch.stack(
            [omega_tilde_PN[2, 1], omega_tilde_PN[0, 2], omega_tilde_PN[1, 0]])

        return position_LP_N, position_LN_N, omega_PN, position_LP_N_unit

    def LLA2PCPF(self,
                 Latitude_Longitude_Altitude_position: Tensor,
                 equatorial_radius: Tensor,
                 polar_radius: Tensor | None = None) -> Tensor:
        """
        Convert from latitude, longitude, altitude to planet-centered, planet-fixed coordinates.
        
        Args:
            Latitude_Longitude_Altitude_position: Tensor containing [latitude, longitude, altitude]
            equatorial_radius: Equatorial radius of the planet
            polar_radius: Polar radius of the planet (None for spherical planet)
            
        Returns:
            Position vector in planet-centered, planet-fixed coordinates
        """
        latitude, longitude, altitude = Latitude_Longitude_Altitude_position[
            0], Latitude_Longitude_Altitude_position[
                1], Latitude_Longitude_Altitude_position[2]

        # For spherical planet (polar_radius is None or < 0)
        planet_eccentricity2 = torch.zeros_like(equatorial_radius).to(
            equatorial_radius.device)
        if polar_radius is not None and polar_radius >= 0:
            planet_eccentricity2 = 1.0 - polar_radius * polar_radius / (
                equatorial_radius * equatorial_radius)

        sin_Phi = torch.sin(latitude)
        N_Val = equatorial_radius / torch.sqrt(1.0 - planet_eccentricity2 *
                                               sin_Phi * sin_Phi)

        x = (N_Val + altitude) * torch.cos(latitude) * torch.cos(longitude)
        y = (N_Val + altitude) * torch.cos(latitude) * torch.sin(longitude)
        z = ((1.0 - planet_eccentricity2) * N_Val + altitude) * sin_Phi

        return torch.stack([x, y, z])

    def DCM_PCPF2SEZ(self, latitude: Tensor, longitude: Tensor) -> Tensor:
        """
        Convert from planet-centered, planet-fixed coordinates to satellite-centered, east-north-up coordinates.
        Based on Basilisk's C_PCPF2SEZ function: Euler2(M_PI_2-lat, m1) * Euler3(longitude, m2)
        
        Args:
            latitude: Latitude in radians
            longitude: Longitude in radians
            
        Returns:
            Direction cosine matrix from planet-fixed to SEZ frame
        """
        # Euler2(M_PI_2 - latitude) - rotation about Y axis
        angle1 = torch.pi / 2.0 - latitude
        direction_cosine_matrix_rotation1 = torch.eye(3,
                                                      dtype=latitude.dtype,
                                                      device=latitude.device)
        direction_cosine_matrix_rotation1[0, 0] = torch.cos(angle1)
        direction_cosine_matrix_rotation1[0, 2] = -torch.sin(angle1)
        direction_cosine_matrix_rotation1[
            2, 0] = -direction_cosine_matrix_rotation1[0, 2]
        direction_cosine_matrix_rotation1[
            2, 2] = direction_cosine_matrix_rotation1[0, 0]

        # Euler3(longitude) - rotation about Z axis
        direction_cosine_matrix_rotation2 = torch.eye(3,
                                                      dtype=longitude.dtype,
                                                      device=longitude.device)
        direction_cosine_matrix_rotation2[0, 0] = torch.cos(longitude)
        direction_cosine_matrix_rotation2[0, 1] = torch.sin(longitude)
        direction_cosine_matrix_rotation2[
            1, 0] = -direction_cosine_matrix_rotation2[0, 1]
        direction_cosine_matrix_rotation2[
            1, 1] = direction_cosine_matrix_rotation2[0, 0]

        return torch.matmul(direction_cosine_matrix_rotation1,
                            direction_cosine_matrix_rotation2)

    def PCPF2LLA(self,
                 Planet_Center_Planet_Fixed_position: Tensor,
                 equatorial_radius: Tensor,
                 polar_radius: Optional[Tensor] = None):
        """
        Convert from planet-centered, planet-fixed coordinates to latitude, longitude, altitude.
        Supports both spherical and ellipsoidal planets.
        Based on Basilisk's PCPF2LLA implementation.
        
        Args:
            Planet_Center_Planet_Fixed_position: Position in planet-centered, planet-fixed coordinates
            equatorial_radius: Equatorial radius of the planet
            polar_radius: Polar radius of the planet (None for spherical planet)
            
        Returns:
            Tensor containing [latitude, longitude, altitude]
        """
        x, y, z = Planet_Center_Planet_Fixed_position[
            0], Planet_Center_Planet_Fixed_position[
                1], Planet_Center_Planet_Fixed_position[2]

        # For spherical planet (polar_radius is None or < 0)
        if polar_radius is None or polar_radius < 0:
            altitude = torch.norm(
                Planet_Center_Planet_Fixed_position
            ) - equatorial_radius  # Altitude is the height above assumed-spherical planet surface
            longitude = torch.atan2(y, x)
            latitude = torch.atan2(z, torch.sqrt(x * x + y * y))
        else:
            # For ellipsoidal planet - following Basilisk's algorithm
            nr_kapp_iter = 10
            good_kappa_acc = 1.0E-13
            planet_eccentricity2 = 1.0 - polar_radius * polar_radius / (
                equatorial_radius * equatorial_radius)
            kappa = 1.0 / (1.0 - planet_eccentricity2)
            p2 = x * x + y * y
            z2 = z * z

            # Iterative solution for kappa
            for i in range(nr_kapp_iter):
                cI = (p2 + (1.0 - planet_eccentricity2) * z2 * kappa * kappa)
                cI = torch.sqrt(
                    cI * cI * cI) / (equatorial_radius * planet_eccentricity2)
                kappaNext = 1.0 + (p2 + (1.0 - planet_eccentricity2) * z2 *
                                   kappa * kappa * kappa) / (cI - p2)

                if torch.abs(kappaNext - kappa) < good_kappa_acc:
                    break
                kappa = kappaNext

            pSingle = torch.sqrt(p2)
            latitude = torch.atan2(kappa * z, pSingle)
            sin_Phi = torch.sin(latitude)
            N_Val = equatorial_radius / torch.sqrt(1.0 - planet_eccentricity2 *
                                                   sin_Phi * sin_Phi)
            altitude = pSingle / torch.cos(latitude) - N_Val
            longitude = torch.atan2(y, x)

        return torch.stack([latitude, longitude, altitude])
