import torch
from torch import Tensor
from typing import NamedTuple


def DCM_PCPF2SEZ(latitude: Tensor, longitude: Tensor) -> Tensor:
    lat = latitude.to(torch.float32)
    lon = longitude.to(torch.float32)

    # 计算旋转角度
    angle1 = torch.pi / 2 - lat  # 对应Euler2的旋转角度 (M_PI_2 - lat)
    angle2 = lon  # 对应Euler3的旋转角度 (longitude)

    # 计算cos和sin值
    cos_angle1 = torch.cos(angle1)
    sin_angle1 = torch.sin(angle1)
    cos_angle2 = torch.cos(angle2)
    sin_angle2 = torch.sin(angle2)

    # 获取batch大小和设备信息
    batch_size = lat.shape[0]
    device = lat.device

    # 构建绕y轴的旋转矩阵 (对应Euler2)
    # 初始化为单位矩阵
    rot2 = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    # 设置非对角线元素
    rot2[:, 0, 0] = cos_angle1
    rot2[:, 0, 2] = -sin_angle1
    rot2[:, 2, 0] = sin_angle1  # 注意C++中是-m[0][2]，而m[0][2]是-sin(x)，所以这里是sin(x)
    rot2[:, 2, 2] = cos_angle1

    # 构建绕z轴的旋转矩阵 (对应Euler3)
    # 初始化为单位矩阵
    rot3 = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    # 设置非对角线元素
    rot3[:, 0, 0] = cos_angle2
    rot3[:, 0, 1] = sin_angle2
    rot3[:, 1, 0] = -sin_angle2
    rot3[:, 1, 1] = cos_angle2

    # 计算矩阵乘积 rot2 * rot3 (使用batch矩阵乘法)
    result = torch.matmul(rot2, rot3)

    return result


def compute_direction_cosine_matrix(latitude: torch.Tensor,
                                    longitude: torch.Tensor) -> torch.Tensor:
    """
    计算经纬度对应的方向余弦矩阵（从ECEF到ENU）
    
    参数:
        latitude: 纬度张量，形状[b]，单位为弧度
        longitude: 经度张量，形状[b]，单位为弧度
    
    返回:
        方向余弦矩阵，形状[b, 3, 3]
    """
    # 计算三角函数值
    sin_phi = torch.sin(latitude)  # [b]
    cos_phi = torch.cos(latitude)  # [b]
    sin_lambda = torch.sin(longitude)  # [b]
    cos_lambda = torch.cos(longitude)  # [b]

    # 构造三行（东、北、天顶）
    row_east = torch.stack(
        [-sin_lambda, cos_lambda,
         torch.zeros_like(sin_lambda)], dim=1)  # [b, 3]
    row_north = torch.stack(
        [-sin_phi * cos_lambda, -sin_phi * sin_lambda, cos_phi],
        dim=1)  # [b, 3]
    row_up = torch.stack([cos_phi * cos_lambda, cos_phi * sin_lambda, sin_phi],
                         dim=1)  # [b, 3]

    # 堆叠为[b, 3, 3]矩阵
    dcm = torch.stack([row_east, row_north, row_up], dim=1)  # [b, 3, 3]

    return dcm


a = torch.rand(1)
b = torch.rand(1)
print(DCM_PCPF2SEZ(a, b))
print(compute_direction_cosine_matrix(a, b))

print(torch.allclose(DCM_PCPF2SEZ(a, b), compute_direction_cosine_matrix(a,
                                                                         b)))
