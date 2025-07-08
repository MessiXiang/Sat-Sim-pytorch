__all__ = [
    'GravityModel',
    'PointMassGravityModel',
    'SphericalHarmonicsGravityModel',
    'PolyhedralGravityModel',
]

from typing import Optional
import torch
from torch import Tensor


class GravityModel:
    """引力模型基类"""
    
    def __init__(self):
        self.mu_body: float = 0.0
        self.rad_equator: float = 0.0
    
    def initialize_parameters(self, grav_body_data) -> Optional[str]:
        """初始化参数
        
        Args:
            grav_body_data: 引力体数据
            
        Returns:
            错误信息，如果成功则返回None
        """
        self.mu_body = grav_body_data.mu
        self.rad_equator = grav_body_data.rad_equator
        return None
    
    def compute_field(self, position_planet_fixed: Tensor) -> Tensor:
        """计算引力场（行星固连系）
        
        Args:
            position_planet_fixed: 行星固连系中的位置向量
            
        Returns:
            引力加速度向量
        """
        raise NotImplementedError
    
    def compute_potential_energy(self, position_wrt_planet_N: Tensor) -> float:
        """计算势能
        
        Args:
            position_wrt_planet_N: 相对于行星的惯性位置向量
            
        Returns:
            势能值
        """
        raise NotImplementedError


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
        
        force_magnitude = -self.mu_body / (r ** 3)
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


class SphericalHarmonicsGravityModel(GravityModel):
    """球谐函数引力模型
    
    基于Pines算法的球谐函数引力计算，支持任意阶数的球谐系数。
    """
    
    def __init__(self):
        super().__init__()
        self.max_degree: int = 0
        self.c_bar: list = []  # 归一化C系数
        self.s_bar: list = []  # 归一化S系数
        
        # 内部计算参数（Pines算法）
        self.a_bar: list = []  # Eq. 61
        self.n1: list = []     # Eq. 63
        self.n2: list = []     # Eq. 64
        self.n_quot1: list = [] # Eq. 79
        self.n_quot2: list = [] # Eq. 80
    
    def _get_k(self, degree: int) -> float:
        """计算项 (2 - d_l)，其中 d_l 是克罗内克δ函数
        
        Args:
            degree: 阶数
            
        Returns:
            k值
        """
        return 1.0 if degree == 0 else 2.0
    
    def initialize_parameters(self, grav_body_data=None) -> Optional[str]:
        """初始化球谐函数参数
        
        Args:
            grav_body_data: 引力体数据（可选）
            
        Returns:
            错误信息，如果成功则返回None
        """
        if grav_body_data:
            self.rad_equator = grav_body_data.rad_equator
            self.mu_body = grav_body_data.mu
        
        if not self.c_bar or not self.s_bar:
            return "Could not initialize spherical harmonics: the 'C' or 'S' parameters were not provided."
        
        # 初始化a_bar矩阵
        self.a_bar = []
        self.n1 = []
        self.n2 = []
        
        for i in range(self.max_degree + 2):
            a_row = [0.0] * (i + 1)
            n1_row = [0.0] * (i + 1)
            n2_row = [0.0] * (i + 1)
            
            # 对角线元素
            if i == 0:
                a_row[i] = 1.0
            else:
                a_row[i] = torch.sqrt(torch.tensor((2 * i + 1) * self._get_k(i) / (2 * i * self._get_k(i - 1)))) * self.a_bar[i - 1][i - 1]
            
            # 计算n1和n2
            for m in range(i + 1):
                if i >= m + 2:
                    n1_row[m] = torch.sqrt(torch.tensor((2 * i + 1) * (2 * i - 1) / ((i - m) * (i + m))))
                    n2_row[m] = torch.sqrt(torch.tensor((i + m - 1) * (2 * i + 1) * (i - m - 1) / ((i + m) * (i - m) * (2 * i - 3))))
            
            self.a_bar.append(a_row)
            self.n1.append(n1_row)
            self.n2.append(n2_row)
        
        # 初始化n_quot1和n_quot2
        self.n_quot1 = []
        self.n_quot2 = []
        
        for l in range(self.max_degree + 1):
            nq1_row = [0.0] * (l + 1)
            nq2_row = [0.0] * (l + 1)
            
            for m in range(l + 1):
                if m < l:
                    nq1_row[m] = torch.sqrt(torch.tensor((l - m) * self._get_k(m) * (l + m + 1) / self._get_k(m + 1)))
                nq2_row[m] = torch.sqrt(torch.tensor((l + m + 2) * (l + m + 1) * (2 * l + 1) * self._get_k(m) / ((2 * l + 3) * self._get_k(m + 1))))
            
            self.n_quot1.append(nq1_row)
            self.n_quot2.append(nq2_row)
        
        return None
    
    def compute_field(self, position_planet_fixed: Tensor, degree: Optional[int] = None, include_zero_degree: bool = True) -> Tensor:
        """计算球谐函数引力场
        
        Args:
            position_planet_fixed: 行星固连系中的位置向量
            degree: 计算的最大阶数（可选，默认使用max_degree）
            include_zero_degree: 是否包含零阶项（点质量项）
            
        Returns:
            引力加速度向量
        """
        if degree is None:
            degree = self.max_degree
        
        if degree > self.max_degree:
            raise ValueError("Requested degree greater than maximum degree in Spherical Harmonics gravity model")
        
        # 提取坐标
        x, y, z = position_planet_fixed[0], position_planet_fixed[1], position_planet_fixed[2]
        
        # 计算球坐标
        r = torch.sqrt(x * x + y * y + z * z)
        if r < 1e-6:
            return torch.zeros(3)
        
        s = x / r  # 方向余弦
        t = y / r
        u = z / r
        
        # 计算a_bar矩阵的对角线以下项
        for l in range(1, degree + 2):
            self.a_bar[l][l - 1] = torch.sqrt(torch.tensor((2 * l) * self._get_k(l - 1) / self._get_k(l))) * self.a_bar[l][l] * u
        
        # 计算a_bar矩阵的其余项
        for m in range(degree + 2):
            for l in range(m + 2, degree + 2):
                self.a_bar[l][m] = u * self.n1[l][m] * self.a_bar[l - 1][m] - self.n2[l][m] * self.a_bar[l - 2][m]
        
        # 计算(2+j*t)^m的实部和虚部
        r_e = [1.0]  # 实部
        i_m = [0.0]  # 虚部
        
        for m in range(1, degree + 2):
            r_e.append(s * r_e[m - 1] - t * i_m[m - 1])
            i_m.append(s * i_m[m - 1] + t * r_e[m - 1])
        
        # 计算rho和rho_l
        rho = self.rad_equator / r
        rho_l = [self.mu_body / r]  # l=0
        
        for l in range(1, degree + 2):
            rho_l.append(rho_l[l - 1] * rho)
        
        # 初始化引力分量
        a1, a2, a3, a4 = 0.0, 0.0, 0.0, 0.0
        
        # 零阶项
        if include_zero_degree:
            a4 = -rho_l[1] / self.rad_equator
        
        # 高阶项
        for l in range(1, degree + 1):
            sum_a1, sum_a2, sum_a3, sum_a4 = 0.0, 0.0, 0.0, 0.0
            
            for m in range(l + 1):
                D = self.c_bar[l][m] * r_e[m] + self.s_bar[l][m] * i_m[m]
                
                if m == 0:
                    E, F = 0.0, 0.0
                else:
                    E = self.c_bar[l][m] * r_e[m - 1] + self.s_bar[l][m] * i_m[m - 1]
                    F = self.s_bar[l][m] * r_e[m - 1] - self.c_bar[l][m] * i_m[m - 1]
                
                sum_a1 += m * self.a_bar[l][m] * E
                sum_a2 += m * self.a_bar[l][m] * F
                
                if m < l:
                    sum_a3 += self.n_quot1[l][m] * self.a_bar[l][m + 1] * D
                
                sum_a4 += self.n_quot2[l][m] * self.a_bar[l + 1][m + 1] * D
            
            a1 += rho_l[l + 1] / self.rad_equator * sum_a1
            a2 += rho_l[l + 1] / self.rad_equator * sum_a2
            a3 += rho_l[l + 1] / self.rad_equator * sum_a3
            a4 -= rho_l[l + 1] / self.rad_equator * sum_a4
        
        return torch.tensor([a1 + s * a4, a2 + t * a4, a3 + u * a4])
    
    def compute_potential_energy(self, position_wrt_planet_N: Tensor) -> float:
        """计算球谐函数势能
        
        注意：当前实现返回点质量近似，完整的球谐函数势能计算较复杂
        
        Args:
            position_wrt_planet_N: 相对于行星的惯性位置向量
            
        Returns:
            势能值
        """
        # TODO: 实现完整的球谐函数势能计算
        # 目前返回点质量近似
        r = torch.norm(position_wrt_planet_N)
        if r < 1e-6:
            return 0.0
        return -self.mu_body / r


class PolyhedralGravityModel(GravityModel):
    """多面体引力模型"""
    
    def __init__(self):
        super().__init__()
        self.vertices: list = []  # 顶点列表
        self.facets: list = []    # 面列表
        self.density: float = 0.0 # 密度
    
    def compute_field(self, position_planet_fixed: Tensor) -> Tensor:
        """计算多面体引力场
        
        Args:
            position_planet_fixed: 行星固连系中的位置向量
            
        Returns:
            引力加速度向量
        """
        # TODO: 实现多面体引力计算
        # 目前返回点质量近似
        r = torch.norm(position_planet_fixed)
        if r < 1e-6:
            return torch.zeros(3)
        
        force_magnitude = -self.mu_body / (r ** 3)
        return force_magnitude * position_planet_fixed
    
    def compute_potential_energy(self, position_wrt_planet_N: Tensor) -> float:
        """计算多面体势能
        
        Args:
            position_wrt_planet_N: 相对于行星的惯性位置向量
            
        Returns:
            势能值
        """
        # TODO: 实现多面体势能计算
        # 目前返回点质量近似
        r = torch.norm(position_wrt_planet_N)
        if r < 1e-6:
            return 0.0
        return -self.mu_body / r
