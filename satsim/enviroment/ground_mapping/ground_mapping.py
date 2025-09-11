__all__ = [
    'GroundMapping',
    'GroundMappingStateDict',
]

from typing import TypedDict
import torch
from satsim.architecture import Module
from satsim.utils import DCM_PCPF2SEZ, PCPF2LLA, mrp_to_rotation_matrix
from satsim.simulation.gravity import Ephemeris
from ..ground_location import AccessState


class GroundMappingStateDict(TypedDict):
    pass


class GroundMapping(Module[GroundMappingStateDict]):

    def __init__(
        self,
        *args,
        minimum_elevation: torch.Tensor,  # [n_sc]
        maximum_range: torch.Tensor | None = None,  # [n_sc]
        half_field_of_view: torch.Tensor,  # [n_sc]
        camera_direction_in_body: torch.Tensor,  # [n_sc,3]
        **kwargs,
    ):
        """This module checks that a vector of mapping points are visible to a spacecraft's imager,
            outputting a vector of accessMessages for each mapping point.

        Args:
            minimum_Elevation (optional): [rad] Minimum elevation angle for a ground mapping point to be visible. Defaults to 0.0.
            maximum_Range (optional): [m] Maximum slant range to compute access for; defaults to -1, which represents no maximum range.
            halfField_Of_View : [rad] Half field of view for the imager. Defaults to 10.0.
            camera_Pos_B (torch.Tensor, optional): [m] Position of the camera in the body frame. Defaults to None, which initializes to zero.
            nHat_B (torch.Tensor, optional): [-] Instrument unit direction vector in body frame components

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
            '_camera_direction_B',
            camera_direction_in_body,
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
    def camera_direction_B(self) -> torch.Tensor:
        return self.get_buffer('_camera_direction_B')

    def forward(
        self,
        state_dict: GroundMappingStateDict | None = None,
        *args,
        ephemeris: Ephemeris,
        spacecraft_position: torch.Tensor,  # [n_sc, 3]
        spacecraft_velocity: torch.Tensor,
        spacecraft_attitude: torch.Tensor,
        locations_in_planet: torch.Tensor,  # [n_p, 3]
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
        Computes ground mapping access and ground state information for a spacecraft and a set of mapping points.
        Args:
            state_dict (GroundMappingStateDict | None, optional): Optional state dictionary for ground mapping.
            *args: Additional positional arguments.
            ephemeris (Ephemeris): Ephemeris data containing planet position and rotation matrices.
            spacecraft_position (torch.Tensor): Spacecraft position in inertial frame.
            spacecraft_velocity (torch.Tensor): Spacecraft velocity in inertial frame.
            spacecraft_attitude (torch.Tensor): Spacecraft attitude quaternion or rotation representation.
            mapping_points (torch.Tensor): Tensor of mapping points in planet-fixed frame, shape [n_sc, n_p, 3].
            **kwargs: Additional keyword arguments.
        Returns:
            tuple:
                - GroundMappingStateDict | None: Updated state dictionary (if any).
                - list[AccessDict]: List of access information dictionaries for each mapping point.
                - list[GroundStateDict]: List of ground state dictionaries for each mapping point.
        Notes:
            - Calculates relative positions, view angles, azimuth, elevation, and access status for each mapping point.
            - Checks instrument field of view and minimum elevation constraints.
            - Intended for use in satellite ground mapping simulations.
        """

        planet_position_in_inertial = ephemeris['position_in_inertial']
        dcm_inertial_to_planetfix = ephemeris['J2000_2_planet_fixed']
        dcm_inertial_to_planetfix_dot = ephemeris['J2000_2_planet_fixed_dot']

        r_BP_N = spacecraft_position - planet_position_in_inertial  # [n_sc, 3]

        dcm_BN = to_rotation_matrix(spacecraft_attitude)

        dcm_NB = dcm_BN.transpose(-2, -1)
        # Get planet frame angular velocity vector
        w_tilde_PN = torch.matmul(
            -dcm_inertial_to_planetfix_dot,
            dcm_inertial_to_planetfix.transpose(-2, -1),
        )
        w_PN_0 = w_tilde_PN[..., 2, 1]
        w_PN_1 = w_tilde_PN[..., 0, 2]
        w_PN_2 = w_tilde_PN[..., 1, 0]
        w_PN = torch.stack([w_PN_0, w_PN_1, w_PN_2], dim=-1)  # [n_sc,3]

        #compute the access of all mapping points

        r_LP_N = torch.matmul(
            dcm_inertial_to_planetfix.transpose(-2, -1),
            locations_in_planet.unsqueeze(-1),
        ).squeeze(-1)  # [n_p, 3]
        rhat_LP_N: torch.Tensor = r_LP_N / torch.norm(
            r_LP_N, dim=-1, keepdim=True)  # [n_p, 3]
        r_LN_N = planet_position_in_inertial + r_LP_N  # [n_p, 3]

        r_BL_N = r_BP_N - r_LP_N.unsqueeze(1)  # [n_p, n_sc, 3]
        BL_distance = torch.norm(r_BL_N, dim=-1,
                                 keepdim=True)  # [n_p, n_sc, 1]
        rhat_BL_N = r_BL_N / BL_distance  # [n_p, n_sc, 3]
        view_angle = torch.asin(
            torch.clamp((rhat_BL_N * rhat_LP_N).sum(dim=-1), -1, 1))

        r_LP_P = locations_in_planet

        lla = PCPF2LLA(
            r_LP_P,
            0.,
            0.,
        )
        lat, longitude, _ = lla.unbind(-1)
        dcm_LP = DCM_PCPF2SEZ(lat, longitude)  # [n_p, 3, 3]

        dcm_LN = torch.matmul(dcm_LP, dcm_inertial_to_planetfix)  # [n_p, 3, 3]
        r_BL_L = torch.matmul(
            dcm_LN.unsqueeze(1),
            r_BL_N.unsqueeze(-1),
        ).squeeze(-1)  # [n_p, n_sc, 3]

        r_BL_L_x, r_BL_L_y, r_BL_L_z = r_BL_L.unbind(-1)
        azimuth = torch.atan2(r_BL_L_y, -r_BL_L_x)

        v_BL_L = torch.matmul(
            dcm_LN.unsqueeze(1),
            spacecraft_velocity -
            torch.cross(w_PN, r_BP_N, dim=-1).unsqueeze(-1),
        ).squeeze(-1)  # [n_p, n_sc, 3]

        range_dot = (v_BL_L * r_BL_L).sum(dim=-1) / BL_distance.squeeze(
            -1)  # [n_p, n_sc, 3]

        v_BL_L_x, v_BL_L_y, v_BL_L_z = v_BL_L.unbind(-1)
        xy_norm = torch.sqrt(r_BL_L_x**2 + r_BL_L_y**2 + 1e-8)
        azimuth_angle_dot = (-r_BL_L_x * v_BL_L_y + r_BL_L_y * v_BL_L_x) / (
            xy_norm**2)  # [n_p, n_sc]

        elevation_dot = (v_BL_L_z / xy_norm - r_BL_L_z *
                         (r_BL_L_x * v_BL_L_x + r_BL_L_y * v_BL_L_y) /
                         xy_norm**3) / (1 + (r_BL_L_y / xy_norm)**2)

        within_view = self._check_instrument_field_of_vision(
            r_LP_N,
            r_BP_N,
            dcm_NB,
        )
        has_access = view_angle > self.minimum_elevation & within_view

        return state_dict, (
            AccessState(
                has_access=has_access,
                slant_range=BL_distance,
                elevation=view_angle,
                azimuth=azimuth,
                range_dot=range_dot,
                azimuth_dot=azimuth_angle_dot,
                elevation_dot=elevation_dot,
                position_BL_L=r_BL_L,
                velocity_BL_L=v_BL_L,
            ),
            r_LN_N,
            r_LP_N,
        )

    def _check_instrument_field_of_vision(
            self,
            r_LP_N: torch.Tensor,  # [n_p, 3]
            r_BP_N: torch.Tensor,  # [n_sc,3]
            dcm_NB: torch.Tensor,  # [n_sc, 3, 3]
    ) -> torch.Tensor:

        # L refer to Location
        # P refer to planet fixed
        # B refer to spacecraft
        # Cp refer to camera direction project

        r_LP_N = r_LP_N.unsqueeze(1)

        r_LB_N = r_LP_N - r_BP_N  #[n_p, n_sc, 3]
        camera_direction_N = torch.matmul(
            dcm_NB,
            self.camera_direction_B.unsqueeze(-1),
        ).squeeze(-1)  # [n_sc, 3]
        camera_normal_proj_distance = (r_LB_N * camera_direction_N).sum(
            dim=-1)  # [n_p, n_sc]

        mask = (camera_normal_proj_distance >= 0) &\
                ((camera_normal_proj_distance <= self.maximum_range)
                | (self.maximum_range < 0)) # [n_p, n_sc]

        view_cone_radius = camera_normal_proj_distance * torch.tan(
            self.half_field_of_view)  # [n_p, n_sc]

        r_CpB_N = camera_normal_proj_distance.unsqueeze(
            -1) * camera_direction_N  # [n_p, n_sc, 3]
        r_LCp_N = torch.norm(r_LB_N - r_CpB_N, dim=-1)

        mask = mask & r_LCp_N < view_cone_radius  # [n_p, n_sc]

        return mask
