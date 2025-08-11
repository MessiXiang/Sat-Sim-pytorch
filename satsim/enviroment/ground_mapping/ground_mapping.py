__all__ = [
    'GroundMapping',
    'GroundMappingStateDict',
    'GroundStateDict',
    'AccessDict',
]

from typing import TypedDict
import torch
from satsim.architecture import Module
from satsim.utils.matrix_support import to_rotation_matrix


class GroundMappingStateDict(TypedDict):
    pass


class GroundStateDict(TypedDict):
    # Position vector of the location with respect to the inertial origin in the inertial frame
    r_LN_N: torch.Tensor
    # Position vector of the location with respect to the planet center in the inertial frame
    r_LP_N: torch.Tensor


class AccessDict(TypedDict):
    has_access: bool  # 1 when the writer has access to a spacecraft; 0 otherwise.
    slant_range: torch.Tensor  # [m] Range from a location to the spacecraft.
    elevation_angle: torch.Tensor  # [rad] Elevation angle for a given spacecraft.
    azimuth_angle: torch.Tensor  # [rad] Azimuth angle for a spacecraft.
    range_dot: torch.Tensor  # [m/s] Range rate of a given spacecraft relative to a location in the SEZ rotating frame.
    elevation_dot: torch.Tensor  # [rad/s] Elevation angle rate for a given spacecraft in the SEZ rotating frame.
    azimuth_angle_dot: torch.Tensor  # [rad/s] Azimuth angle rate for a given spacecraft in the SEZ rotating frame.
    r_BL_L: torch.Tensor  # [m] Spacecraft position relative to the groundLocation in the SEZ frame.
    v_BL_L: torch.Tensor  # [m/s] SEZ relative time derivative of r_BL vector in SEZ vector components.


class GroundMapping(Module[GroundMappingStateDict]):

    def __init__(self,
                 *args,
                 minimum_elevation: torch.Tensor | None = None,
                 maximum_range: torch.Tensor | None = None,
                 half_field_of_view: torch.Tensor | None = None,
                 camera_pos_in_body: torch.Tensor | None = None,
                 camera_direction_in_body: torch.Tensor | None = None,
                 **kwargs):
        """This module checks that a vector of mapping points are visible to a spacecraft's imager,
            outputting a vector of accessMessages for each mapping point.

        Args:
            minimum_Elevation (optional): [rad] Minimum elevation angle for a mapping point to be visible. Defaults to 0.0.
            maximum_Range (optional): [m] Maximum slant range to compute access for; defaults to -1, which represents no maximum range.
            halfField_Of_View : [rad] Half field of view for the imager. Defaults to 10.0.
            camera_Pos_B (torch.Tensor, optional): [m] Position of the camera in the body frame. Defaults to None, which initializes to zero.
            nHat_B (torch.Tensor, optional): [-] Instrument unit direction vector in body frame components

        """
        super().__init__(*args, **kwargs)

        minimum_elevation = torch.tensor(
            [0.0 * torch.pi / 180], dtype=torch.float32
        ) if minimum_elevation is None else minimum_elevation

        maximum_range = torch.tensor(
            [-1],
            dtype=torch.float32) if maximum_range is None else maximum_range

        half_field_of_view = torch.tensor(
            [10.0 * torch.pi / 180], dtype=torch.float32
        ) if half_field_of_view is None else half_field_of_view

        #
        camera_pos_in_body = torch.zeros(
            3, dtype=torch.float32
        ) if camera_pos_in_body is None else camera_pos_in_body

        #
        camera_direction_in_body = torch.zeros(
            3, dtype=torch.float32
        ) if camera_direction_in_body is None else camera_direction_in_body

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
            '_camera_pos_in_body',
            camera_pos_in_body,
            persistent=False,
        )
        self.register_buffer(
            '_camera_direction_in_body',
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
    def camera_pos_in_body(self) -> torch.Tensor:
        return self.get_buffer('_camera_pos_in_body')

    @property
    def camera_direction_in_body(self) -> torch.Tensor:
        return self.get_buffer('_camera_direction_in_body')

    def forward(
        self,
        state_dict: GroundMappingStateDict | None = None,
        planet_Position_in_inertial: torch.Tensor | None = None,
        dcm_inertial_to_PlanetFix: torch.Tensor | None = None,
        dcm_inertial_to_PlanetFix_dot: torch.Tensor | None = None,
        r_BN_N: torch.Tensor | None = None,
        v_BN_N: torch.Tensor | None = None,
        sigma_BN: torch.Tensor | None = None,
        mapping_Points: list[torch.Tensor]
        | None = None,  #list of mapping points in the inertial frame
        *args,
        **kwargs,
    ) -> tuple[
            GroundMappingStateDict | None,
            list[AccessDict],
            list[GroundStateDict],
    ]:
        """This is the main method that gets called every time the module is updated.
            Args:
                state_dict (GroundMappingStateDict | None): The state dictionary of the module.
                dcm_J2000_to_PlanetFix: torch.Tensor | None: Orientation matrix of planet-fixed relative to inertial
                dcm_J2000_to_PlanetFix_dot: torch.Tensor | None: Time derivative of the orientation matrix of planet-fixed relative to inertial
                r_BN_N: torch.Tensor = None: The position vector of the spacecraft in the inertial frame.
                v_BN_N: torch.Tensor | None = None: The velocity vector of the spacecraft in the inertial frame.
                sigma_BN: torch.Tensor | None = None: The attitude of the spacecraft in the inertial frame.
                mapping_Points: list[torch.Tensor] | None = None: The mapping points in the inertial frame.

        """

        #Initialize lists to store access information and current ground states.They are all output parameters
        accessDict: list[AccessDict] = []
        currentGroundState: list[GroundStateDict] = []

        #initialize the v_BN_N vector
        if v_BN_N is not None:
            v_BN_N = v_BN_N
        else:
            v_BN_N = torch.zeros_like(r_BN_N)

        #update the relevant rotations matrices --- updateInertialPositions
        if dcm_inertial_to_PlanetFix is not None:
            dcm_PN = dcm_inertial_to_PlanetFix
        else:
            dcm_PN = torch.eye(3, device=r_BN_N.device, dtype=r_BN_N.dtype)

        if dcm_inertial_to_PlanetFix_dot is not None:
            dcm_PN_dot = dcm_inertial_to_PlanetFix_dot
        else:
            dcm_PN_dot = torch.zeros_like(dcm_PN)

        if planet_Position_in_inertial is not None:
            r_PN_N = planet_Position_in_inertial
        else:
            r_PN_N = torch.zeros_like(r_BN_N)

        r_BP_N = r_BN_N - r_PN_N

        dcm_BN = to_rotation_matrix(sigma_BN)

        dcm_NB = dcm_BN.transpose(-2, -1)
        # Get planet frame angular velocity vector
        w_tilde_PN = torch.matmul(-dcm_PN_dot, dcm_PN.transpose(-2, -1))
        w_PN_0 = w_tilde_PN[..., 2, 1]
        w_PN_1 = w_tilde_PN[..., 0, 2]
        w_PN_2 = w_tilde_PN[..., 1, 0]
        w_PN = torch.stack([w_PN_0, w_PN_1, w_PN_2], dim=-1)

        #compute the access of all mapping points
        for i in range(len(mapping_Points)):
            #computeAccess
            r_LP_N = torch.matmul(dcm_PN.transpose(-2, -1),
                                  mapping_Points[i].unsqueeze(-1)).squeeze(-1)
            rhat_LP_N = r_LP_N / torch.linalg.norm(
                r_LP_N, dim=-1, keepdim=True)
            r_LN_N = r_PN_N + r_LP_N

            currentGroundState.append(
                GroundStateDict(r_LN_N=r_LN_N, r_LP_N=r_LP_N))

            r_BL_N = r_BP_N - r_LP_N
            r_BL_mag = torch.linalg.norm(r_BL_N, dim=-1, keepdim=True)
            relative_Heading_N = r_BL_N / r_BL_mag
            viewAngle = (torch.pi / 2 - torch.acos(
                torch.clamp(
                    (relative_Heading_N * rhat_LP_N).sum(dim=-1), -1, 1)))

            # Compute dcm_LP from r_LP_N and dcm_inertial_to_PlanetFix
            dcm_LP = compute_dcm_LP_from_r_LP_N(
                r_LP_N=r_LP_N,
                dcm_inertial_to_PlanetFix=dcm_inertial_to_PlanetFix)

            ### Compute the items that are needed for the access dictionary
            r_BL_L = torch.matmul(torch.matmul(dcm_LP, dcm_PN),
                                  r_BL_N.unsqueeze(-1)).squeeze(-1)

            #azimuth shape:[...,1]
            azimuth = torch.atan2(
                (r_BL_L[..., 1] /
                 (torch.sqrt(r_BL_L[..., 0]**2 + r_BL_L[..., 1]**2 +
                             1e-8))).unsqueeze(-1),
                (-r_BL_L[..., 0] /
                 (torch.sqrt(r_BL_L[..., 0]**2 + r_BL_L[..., 1]**2 +
                             1e-8))).unsqueeze(-1))

            v_BL_L = torch.matmul(
                torch.matmul(dcm_LP, dcm_PN), v_BN_N -
                torch.cross(w_PN, r_BP_N, dim=-1).unsqueeze(-1)).squeeze(-1)

            range_dot = (v_BL_L * r_BL_L).sum(dim=-1) / r_BL_mag

            xy_norm = torch.sqrt(r_BL_L[..., 0]**2 + r_BL_L[..., 1]**2 + 1e-8)
            azimuth_angle_dot = (-r_BL_L[..., 0] * v_BL_L[..., 1] +
                                 r_BL_L[..., 1] * v_BL_L[..., 0]) / (xy_norm**
                                                                     2)
            elevation_dot = (v_BL_L[..., 2] / xy_norm - r_BL_L[..., 2] *
                             (r_BL_L[..., 0] * v_BL_L[..., 0] +
                              r_BL_L[..., 1] * v_BL_L[..., 1]) /
                             xy_norm**3) / (1 + (r_BL_L[..., 2] / xy_norm)**2)

            within_view = _check_instrument_fieldofvision(
                r_LP_N, r_BP_N, dcm_NB, self.camera_pos_in_body,
                self.camera_direction_in_body, self.maximum_range,
                self.half_field_of_view)

            if (viewAngle > self.minimum_elevation and within_view):
                has_Access = True
            else:
                has_Access = False

            accessDict.append(
                AccessDict(
                    slant_range=r_BL_mag,
                    elevation_angle=viewAngle,
                    r_BL_L=r_BL_L,
                    azimuth=azimuth,
                    v_BL_L=v_BL_L,
                    range_dot=range_dot,
                    azimuth_angle_dot=azimuth_angle_dot,
                    elevation_dot=elevation_dot,
                    has_access=has_Access,
                ))

        return state_dict, (accessDict, currentGroundState)


def _check_instrument_fieldofvision(r_LP_N: torch.Tensor, r_BP_N: torch.Tensor,
                                    dcm_NB: torch.Tensor,
                                    cameraPos_B: torch.Tensor,
                                    nHat_B: torch.Tensor, maximum_Range: float,
                                    half_FieldOfView: float) -> bool:

    #源代码中列向量乘行向量 出来的boresightNormalProj是个标量
    boresight_Normal_Proj_mul_1 = (
        r_LP_N -
        (r_BP_N + torch.matmul(dcm_NB, cameraPos_B.unsqueeze(-1)).squeeze(-1)))
    boresight_Normal_Proj_mul_2 = torch.matmul(
        dcm_NB, nHat_B.unsqueeze(-1)).squeeze(-1)
    boresight_Normal_Proj = (boresight_Normal_Proj_mul_1 *
                             boresight_Normal_Proj_mul_2).sum(dim=-1)

    if (boresight_Normal_Proj >= 0) and (boresight_Normal_Proj <= maximum_Range
                                         or maximum_Range < 0):
        coneRadius = boresight_Normal_Proj * torch.tan(half_FieldOfView)

        orthDistance_cal_1 = r_LP_N - (r_BP_N + torch.matmul(
            dcm_NB, cameraPos_B.unsqueeze(-1)).squeeze(-1))
        orthDistance_cal_2 = torch.matmul(boresight_Normal_Proj * dcm_NB,
                                          nHat_B.unsqueeze(-1)).squeeze(-1)
        orthDistance = torch.linalg.norm(orthDistance_cal_1 -
                                         orthDistance_cal_2,
                                         dim=-1)

        if orthDistance <= coneRadius:
            return True
    return False


def compute_dcm_LP_from_r_LP_N(
        r_LP_N: torch.Tensor,
        dcm_inertial_to_PlanetFix: torch.Tensor) -> torch.Tensor:
    """
    根据地面站在惯性系中的位置和地固系的姿态，计算 dcm_LP(从地固系到地面站局部SEZ系的旋转矩阵)

    参数:
        r_LP_N (torch.Tensor): shape (3,) 地面站在惯性系 N 中的位置向量 [m]
        dcm_J2000_to_PlanetFix (torch.Tensor): shape (3, 3) 惯性系 N 到地固系 P 的方向余弦矩阵

    返回:
        dcm_LP (torch.Tensor): shape (3, 3) 从地固系 P 到地面站局部坐标系 L (SEZ) 的旋转矩阵
    """

    # 将地面站位置从惯性系转换到地固系
    r_LP_P = torch.matmul(dcm_inertial_to_PlanetFix,
                          r_LP_N.unsqueeze(-1)).squeeze(-1)

    # 计算纬度和经度
    x, y, z = r_LP_P
    r_xy = torch.sqrt(x**2 + y**2)
    lon = torch.atan2(y, x)
    lat = torch.atan2(z, r_xy)

    # 构造方向余弦矩阵 dcm_LP
    sinLat = torch.sin(lat)
    cosLat = torch.cos(lat)
    sinLon = torch.sin(lon)
    cosLon = torch.cos(lon)

    dcm_LP = torch.stack([
        torch.stack([-sinLat * cosLon, -sinLat * sinLon, cosLat], dim=-1),
        torch.stack([-sinLon, cosLon,
                     torch.tensor(0.0, dtype=lat.dtype)],
                    dim=-1),
        torch.stack([-cosLat * cosLon, -cosLat * sinLon, -sinLat], dim=-1)
    ],
                         dim=-2)

    return dcm_LP
