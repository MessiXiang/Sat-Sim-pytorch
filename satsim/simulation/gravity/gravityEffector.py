__all__ = [
    'GravityEffector', 'GravBodyData', 'PlanetState',
    'GravityEffectorStateDict', 'GravBodyDataStateDict'
]

from dataclasses import dataclass
from typing import List, Optional
import torch
from torch import Tensor
from typing_extensions import TypedDict

from .gravityModel import GravityModel, PointMassGravityModel
from satsim.architecture import Module, Timer


@dataclass
class PlanetBodyStateDict(TypedDict):
    """行星状态数据结构"""
    position_vector: Tensor
    velocity_vector: Tensor
    J2000_2_planet_fixed: Tensor
    J2000_2_planet_fixed_dot: Tensor


@dataclass
class GravBodyData:
    """引力体数据类
    """
    mu: float = 0.0
    rad_equator: float = 0.0
    radius_ratio: float = 1.0
    gravity_model: Optional[GravityModel] = None

    # 引力模型
    gravity_model = gravity_model or PointMassGravityModel()

    local_planet: PlanetBodyStateDict = PlanetBodyStateDict(
        position_vector=torch.zeros(3),
        velocity_vector=torch.zeros(3),
        J2000_2_planet_fixed=torch.eye(3),
        J2000_2_planet_fixed_dot=torch.zeros(3, 3),
    )

    def compute_gravity_inertial(self,
                                 r_inertial: Tensor,
                                 dt: float = 0.0) -> Tensor:
        """计算惯性系中的引力加速度

        Args:
            r_inertial: 惯性位置向量
            dt: 时间步长
        Returns:
            引力加速度向量
        """
        # 更新姿态矩阵
        direction_cos_matrix_inertial_2_planet_fixed = self.local_planet.J2000_2_planet_fixed
        if torch.allclose(direction_cos_matrix_inertial_2_planet_fixed,
                          torch.zeros(3, 3)):
            direction_cos_matrix_inertial_2_planet_fixed = torch.eye(3)

        direction_cos_matrix_inertial_2_planet_fixed_dot = self.local_planet.J2000_2_planet_fixed_dot
        direction_cos_matrix_inertial_2_planet_fixed = direction_cos_matrix_inertial_2_planet_fixed + direction_cos_matrix_inertial_2_planet_fixed_dot * dt

        # 用赋值保持梯度
        self.local_planet.J2000_2_planet_fixed = direction_cos_matrix_inertial_2_planet_fixed
        self.local_planet.J2000_2_planet_fixed_dot = direction_cos_matrix_inertial_2_planet_fixed_dot

        # 计算行星固连系中的位置
        direction_cos_matrix_planet_fixed_2_inertial = direction_cos_matrix_inertial_2_planet_fixed.T
        r_planet_fixed = direction_cos_matrix_planet_fixed_2_inertial @ r_inertial

        # 计算引力场
        grav_planet_fixed = self.gravity_model.compute_field(r_planet_fixed)

        # 转换回惯性系
        return direction_cos_matrix_inertial_2_planet_fixed @ grav_planet_fixed


# 需要吗
class GravityEffectorStateDict(TypedDict):
    """引力效应器状态字典"""
    grav_bodies: List[GravBodyData]  # 引力体列表
    central_body: Optional[GravBodyData]  # 中心天体


class GravityEffector(Module[GravityEffectorStateDict]):
    """引力效应器类
    """

    def __init__(
        self,
        timer: Timer,
        grav_bodies: List[GravBodyData],
        central_body: GravBodyData | None = None,
    ):
        super().__init__()

        self.grav_bodies = grav_bodies or []
        self.central_body: Optional[GravBodyData] = central_body

    def reset(self) -> GravityEffectorStateDict:
        """重置状态"""
        state_dict = super().reset()

        # 初始化所有引力体
        for body in self.grav_bodies:
            body.gravity_model.initialize_parameters(body)

        state_dict.update({
            'grav_bodies': self.grav_bodies,
            'central_body': self.central_body,
        })

        return state_dict

    def forward(
        self,
        state_dict,
        *args,
        grav_bodies_ephemerics: list[PlanetBodyStateDict],
        central_body_ephemeric: PlanetBodyStateDict | None,
        **kwargs,
    ) -> PlanetBodyStateDict:
        # 重置中心天体
        central_body = state_dict['central_body']
        grav_bodies = state_dict['grav_bodies']

        # 更新所有引力体
        for body in self.grav_bodies:
            # body.load_ephemeris()

            if not body.is_central_body:
                continue

            if central_body:
                raise ValueError(
                    "Specified two central bodies at the same time")
            else:
                central_body = body

        return self.central_body.local_planet

    def compute_gravity_field(
            self, r_spacecraft2frame_inertial: Tensor,
            rDot_spacecraft2frame_inertial: Tensor) -> Tensor:
        """计算引力场
        
        Args:
            r_spacecraft2frame_inertial: 航天器相对于参考系的位置向量 (spacecraft relative to frame, in inertial frame)
            rDot_spacecraft2frame_inertial: 航天器相对于参考系的速度向量 (spacecraft velocity relative to frame, in inertial frame)
            
        Returns:
            rDotDot_spacecraft2frame_inertial: 航天器相对于参考系的引力加速度向量
        """
        if self.time_corr is None:
            system_clock = 0.0
        else:
            system_clock = float(self.time_corr[0])

        # 确定参考系
        if self.central_body:
            r_center2inertial_inertial = self._get_euler_stepped_grav_body_position(
                self.central_body)
            r_spacecraft2inertial_inertial = r_spacecraft2frame_inertial + r_center2inertial_inertial
        else:
            r_spacecraft2inertial_inertial = r_spacecraft2frame_inertial

        # 计算总引力加速度
        rDotDot_spacecraft2frame_inertial = torch.zeros(3)

        for body in self.grav_bodies:
            # 计算行星位置
            r_planet2inertial_inertial = self._get_euler_stepped_grav_body_position(
                body)
            r_spacecraft2planet_inertial = r_spacecraft2inertial_inertial - r_planet2inertial_inertial

            # 处理相对动力学
            if self.central_body and not body.is_central_body:
                # 减去中心天体由于其他天体产生的加速度
                rDotDot_spacecraft2frame_inertial = rDotDot_spacecraft2frame_inertial + body.compute_gravity_inertial(
                    r_planet2inertial_inertial - r_center2inertial_inertial,
                    self._timer.dt)

            # 计算引力加速度
            rDotDot_spacecraft2frame_inertial = rDotDot_spacecraft2frame_inertial + body.compute_gravity_inertial(
                r_spacecraft2planet_inertial, self._timer.dt)

        return rDotDot_spacecraft2frame_inertial

    # input -> output
    def update_inertial_position_and_velocity(
            self, r_spacecraft2frame_inertial: Tensor,
            rDot_spacecraft2frame_inertial: Tensor) -> None:
        """更新惯性位置和速度
        
        Args:
            r_spacecraft2frame_inertial: 航天器相对于参考系的位置向量 (spacecraft relative to frame, in inertial frame)
            rDot_spacecraft2frame_inertial: 航天器相对于参考系的速度向量 (spacecraft velocity relative to frame, in inertial frame)
        """
        if self.central_body:
            r_center2inertial_inertial = self._get_euler_stepped_grav_body_position(
                self.central_body)
            inertial_position_property = r_center2inertial_inertial + r_spacecraft2frame_inertial
            inertial_velocity_property = self.central_body.local_planet.velocity_vector + rDot_spacecraft2frame_inertial
        else:
            inertial_position_property = r_spacecraft2frame_inertial
            inertial_velocity_property = rDot_spacecraft2frame_inertial

        return inertial_position_property, inertial_velocity_property

    def _get_euler_stepped_grav_body_position(
            self, body_data: GravBodyData) -> Tensor:
        """使用欧拉积分计算行星位置"""
        # if self.time_corr is None:
        #     system_clock = 0.0
        # else:
        #     system_clock = float(self.time_corr[0])
        # dt = system_clock - body_data.time_written

        dt = self.timer.dt
        r_planet2inertial_inertial = body_data.local_planet.position_vector + body_data.local_planet.velocity_vector * dt

        return r_planet2inertial_inertial

    def update_energy_contributions(
            self, r_spacecraft2frame_inertial: Tensor) -> float:
        """计算势能贡献
        
        Args:
            r_spacecraft2frame_inertial: 航天器相对于参考系的位置向量
            
        Returns:
            orbit_potential_energy_contribution: 轨道势能贡献
        """
        # if self.time_corr is None:
        #     system_clock = 0.0
        # else:
        #     system_clock = float(self.time_corr[0])

        # 确定参考系
        if self.central_body:
            r_center2inertial_inertial = self._get_euler_stepped_grav_body_position(
                self.central_body)
            r_spacecraft2inertial_inertial = r_spacecraft2frame_inertial + r_center2inertial_inertial
        else:
            r_spacecraft2inertial_inertial = r_spacecraft2frame_inertial  # frame为惯性系

        orbit_potential_energy_contribution = torch.tensor(0.)

        for body in self.grav_bodies:
            r_planet2inertial_inertial = self._get_euler_stepped_grav_body_position(
                body)
            r_spacecraft2planet_inertial = r_spacecraft2inertial_inertial - r_planet2inertial_inertial

            if self.central_body and not body.is_central_body:
                # 中心天体在当前行星场中的势能（相对动力学修正）
                r_planet2center_inertial = r_planet2inertial_inertial - r_center2inertial_inertial
                orbit_potential_energy_contribution = orbit_potential_energy_contribution + body.gravity_model.compute_potential_energy(
                    r_planet2center_inertial)

            # 航天器在当前行星场中的势能
            orbit_potential_energy_contribution = orbit_potential_energy_contribution + body.gravity_model.compute_potential_energy(
                r_spacecraft2planet_inertial)

        return orbit_potential_energy_contribution
