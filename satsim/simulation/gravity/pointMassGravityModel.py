__all__ = [
    'PointMassGravityModel',
]

import torch
from torch import Tensor
from satsim.simulation.gravity.gravityModel import GravityModel


class PointMassGravityModel(GravityModel):
    """点质量引力模型"""

    def compute_field(self, position_planet_fixed: Tensor) -> Tensor:
        """计算点质量引力场
        
        Args:
            position_planet_fixed: 行星固连系中的位置向量
            
        Returns:
            引力加速度向量
        """
        r = torch.norm(position_planet_fixed)
        if r < 1e-6:
            return torch.zeros(3)

        force_magnitude = -self.mu_body / (r**3)
        return force_magnitude * position_planet_fixed

    def compute_potential_energy(self, position_wrt_planet_N: Tensor) -> float:
        """计算点质量势能
        
        Args:
            position_wrt_planet_N: 相对于行星的惯性位置向量
            
        Returns:
            势能值
        """
        r = torch.norm(position_wrt_planet_N)
        if r < 1e-6:
            return 0.0
        return -self.mu_body / r
