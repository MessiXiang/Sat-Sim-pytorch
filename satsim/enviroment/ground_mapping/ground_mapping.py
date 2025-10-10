__all__ = [
    'GroundMapping',
    'GroundMappingStateDict',
]

from typing import TypedDict

import torch
import torch.nn.functional as F

from satsim.architecture import Module
from satsim.simulation.gravity import Ephemeris
from satsim.utils import DCM_PCPF2SEZ, PCPF2LLA, mrp_to_rotation_matrix

from ..ground_location import AccessState


class GroundMappingStateDict(TypedDict):
    pass


class GroundMapping(Module[GroundMappingStateDict]):
    """Ground mapping module for satellite imaging simulations.

    This module evaluates visibility and access conditions for multiple ground mapping points
    from a spacecraft's imager, considering instrument field of view, elevation constraints,
    and range limitations.
    """

    def __init__(
        self,
        *args,
        minimum_elevation: torch.Tensor,  # [n_sc]
        maximum_range: torch.Tensor | None = None,  # [n_sc]
        half_field_of_view: torch.Tensor,  # [n_sc]
        camera_direction_B_B: torch.Tensor,  # [n_sc,3]
        **kwargs,
    ):
        """
        This module checks whether a set of ground mapping points are visible to a spacecraft's imager,
        and outputs an access message vector for each mapping point.

        Args:
            minimum_elevation (optional): [rad] Minimum elevation angle for a ground mapping point to be considered visible. Default is 0.0.
            maximum_range (optional): [m] Maximum slant range for access calculation; default is -1, meaning no maximum range.
            half_field_of_view: [rad] Half field of view angle of the imager. Default is 10.0.
            camera_direction_B_B (torch.Tensor): [n_sc, 3] Camera direction vector in the body frame.
        """
        super().__init__(*args, **kwargs)

        maximum_range = torch.tensor(
            [-1.]) if maximum_range is None else maximum_range

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
        self.register_buffer(
            '_half_field_of_view',
            half_field_of_view,
            persistent=False,
        )
        self.register_buffer(
            '_camera_direction_B_B',
            camera_direction_B_B,
            persistent=False,
        )

    @property
    def minimum_elevation(self) -> torch.Tensor:
        return self.get_buffer('_minimum_elevation')

    @property
    def maximum_range(self) -> torch.Tensor:
        return self.get_buffer('_maximum_range')

    @property
    def half_field_of_view(self) -> torch.Tensor:
        return self.get_buffer('_half_field_of_view')

    @property
    def camera_direction_B_B(self) -> torch.Tensor:
        return self.get_buffer('_camera_direction_B_B')

    def forward(
        self,
        state_dict: GroundMappingStateDict | None = None,
        *args,
        ephemeris: Ephemeris,
        position_BN_N: torch.Tensor,  # [n_sc, 3]
        velocity_BN_N: torch.Tensor,
        attitude_BN: torch.Tensor,
        position_LP_P: torch.Tensor,  # [n_p, 3] - p stands for mapping point
        equatorial_radius: float,
        polar_radius: float,
        **kwargs,
    ) -> tuple[
            GroundMappingStateDict,
            tuple[
                AccessState,
                torch.Tensor,
                torch.Tensor,
            ],
    ]:
        """
        Calculate ground mapping access and ground state information for a spacecraft and a set of mapping points.

        Args:
            state_dict (GroundMappingStateDict | None, optional): Optional state dictionary for ground mapping.
            *args: Additional positional arguments.
            ephemeris (Ephemeris): Ephemeris data containing planet position and rotation matrices.
            position_BN_N (torch.Tensor): Spacecraft position in the inertial frame.
            velocity_BN_N (torch.Tensor): Spacecraft velocity in the inertial frame.
            attitude_BN (torch.Tensor): Spacecraft attitude (quaternion or rotation matrix).
            position_LP_P (torch.Tensor): Mapping points in planet-fixed frame, shape [n_p, 3].
            equatorial_radius (float): Equatorial radius of the planet.
            polar_radius (float): Polar radius of the planet.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple:
                - GroundMappingStateDict | None: Updated state dictionary (if any).
                - tuple:
                    - AccessState: Access information for each mapping point.
                    - torch.Tensor: Additional output tensor 1.
                    - torch.Tensor: Additional output tensor 2.

        Notes:
            - Computes relative positions, view angles, azimuth, elevation, and access status for each mapping point.
            - Checks instrument field of view and minimum elevation constraints.
            - Designed for satellite ground mapping simulation scenarios.
        """

        position_PN_N = ephemeris['position_CN_N']
        direction_cosine_matrix_PN = ephemeris['direction_cosine_matrix_CN']
        direction_cosine_matrix_PN_dot = ephemeris[
            'direction_cosine_matrix_CN_dot']

        position_BP_N = position_BN_N - position_PN_N  # [n_sc, 3]

        direction_cosine_matrix_BN = mrp_to_rotation_matrix(attitude_BN)

        direction_cosine_matrix_NB = direction_cosine_matrix_BN.transpose(
            -2, -1)
        # Get planet frame angular velocity vector
        angular_velocity_tilde_PN = torch.einsum(
            '...ij,...kj->...ik',
            -direction_cosine_matrix_PN_dot,
            direction_cosine_matrix_PN,
        )

        angular_velocity_PN_0 = angular_velocity_tilde_PN[..., 2, 1]
        angular_velocity_PN_1 = angular_velocity_tilde_PN[..., 0, 2]
        angular_velocity_PN_2 = angular_velocity_tilde_PN[..., 1, 0]
        angular_velocity_PN = torch.stack([
            angular_velocity_PN_0, angular_velocity_PN_1, angular_velocity_PN_2
        ],
                                          dim=-1)  # [n_sc,3]

        #compute the access of all mapping points

        position_LP_N = torch.einsum('...ji,...j->...i',
                                     direction_cosine_matrix_PN,
                                     position_LP_P)  # [n_p, 3]

        position_LP_N_unit: torch.Tensor = F.normalize(
            position_LP_N)  # [n_p, 3]
        position_LN_N = position_PN_N + position_LP_N  # [n_p, 3]

        position_BL_N = position_BP_N.unsqueeze(0) - position_LP_N.unsqueeze(
            1)  # [n_p, n_sc, 3]
        position_BL_N_norm = torch.norm(position_BL_N, dim=-1,
                                        keepdim=True)  # [n_p, n_sc, 1]
        position_BL_N_unit = position_BL_N / position_BL_N_norm  # [n_p, n_sc, 3]
        sin_view_angle = torch.einsum(
            '...i, ...ji -> ...j',
            position_LP_N_unit,
            position_BL_N_unit,
        )
        view_angle = torch.asin(torch.clamp(
            sin_view_angle,
            -1,
            1,
        ))

        lla = PCPF2LLA(
            position_LP_P,
            equatorial_radius=equatorial_radius,
            polar_radius=polar_radius,
        )

        latitude, longitude, _ = lla.unbind(-1)
        direction_cosine_matrix_LP = DCM_PCPF2SEZ(latitude,
                                                  longitude)  # [n_p, 3, 3]

        direction_cosine_matrix_LN = torch.einsum(
            '...ik,...kj->...ij', direction_cosine_matrix_LP,
            direction_cosine_matrix_PN)  # [n_p, 3, 3]
        position_BL_L = torch.einsum(
            '...ij,...kj->...ki',
            direction_cosine_matrix_LN,  # [n_p, 3, 3]
            position_BL_N,  # [n_p, n_sc, 3]
        )  # [n_p, n_sc, 3]

        position_BL_L_x, position_BL_L_y, position_BL_L_z = position_BL_L.unbind(
            -1)
        azimuth = torch.atan2(position_BL_L_y, -position_BL_L_x)

        velocity_BL_L = torch.einsum(
            '...ij,...kj->...ki',
            direction_cosine_matrix_LN,  # [n_p, 3, 3]
            (velocity_BN_N -
             torch.cross(angular_velocity_PN, position_BP_N,
                         dim=-1)).unsqueeze(0),  # [1, n_sc, 3]
        )  # [n_p, n_sc, 3]

        range_dot = (velocity_BL_L * position_BL_L).sum(
            dim=-1) / position_BL_N_norm.squeeze(-1)  # [n_p, n_sc, 3]

        velocity_BL_L_x, velocity_BL_L_y, velocity_BL_L_z = velocity_BL_L.unbind(
            -1)
        xy_norm = torch.sqrt(position_BL_L_x**2 + position_BL_L_y**2 + 1e-8)
        azimuth_angle_dot = (-position_BL_L_x * velocity_BL_L_y +
                             position_BL_L_y * velocity_BL_L_x) / (
                                 xy_norm**2)  # [n_p, n_sc]

        elevation_dot = (velocity_BL_L_z / xy_norm - position_BL_L_z *
                         (position_BL_L_x * velocity_BL_L_x +
                          position_BL_L_y * velocity_BL_L_y) / xy_norm**3) / (
                              1 + (position_BL_L_y / xy_norm)**2)

        within_view = self._check_instrument_field_of_vision(
            position_LP_N,
            position_BP_N,
            direction_cosine_matrix_NB,
        )
        has_access = (view_angle > self.minimum_elevation) & within_view
        return state_dict, (
            AccessState(
                has_access=has_access,
                slant_range=position_BL_N_norm,
                elevation=view_angle,
                azimuth=azimuth,
                range_dot=range_dot,
                azimuth_dot=azimuth_angle_dot,
                elevation_dot=elevation_dot,
                position_BL_L=position_BL_L,
                velocity_BL_L=velocity_BL_L,
            ),
            position_LN_N,
            position_LP_N,
        )

    def _check_instrument_field_of_vision(
        self,
        position_LP_N: torch.Tensor,
        position_BP_N: torch.Tensor,
        direction_cosine_matrix_NB: torch.Tensor,
    ) -> torch.Tensor:

        position_LP_N = position_LP_N.unsqueeze(1)  # [n_p, 1, 3]

        position_LB_N = position_LP_N - position_BP_N.unsqueeze(
            0)  # [n_p, n_sc, 3]
        camera_direction_B_N = torch.einsum(
            '...ij,...j->...i',
            direction_cosine_matrix_NB,  # [n_sc, 3, 3]
            self.camera_direction_B_B,  # [n_sc, 3]
        )  # [n_sc, 3]
        camera_normal_proj_distance = torch.einsum(
            '...j,...j->...',
            position_LB_N,
            camera_direction_B_N,
        )  # [n_p, n_sc]

        mask = ((camera_normal_proj_distance >= 0) &
                ((camera_normal_proj_distance <= self.maximum_range)
                 | (self.maximum_range < 0)))  # [n_p, n_sc]

        view_cone_radius = camera_normal_proj_distance * torch.tan(
            self.half_field_of_view)  # [n_p, n_sc]

        position_CpB_N = camera_normal_proj_distance.unsqueeze(
            -1) * camera_direction_B_N  # [n_p, n_sc, 3]
        position_LCp_N_norm = torch.norm(position_LB_N - position_CpB_N,
                                         dim=-1)

        mask = mask & (position_LCp_N_norm < view_cone_radius)  # [n_p, n_sc]

        return mask
