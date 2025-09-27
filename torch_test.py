import torch

from satsim.utils import mrp_to_rotation_matrix


def _DCM_to_MRP(dcm: torch.Tensor) -> torch.Tensor:
    """
    Convert a direction cosine matrix to a MRP vector. 
    Input:
        dcm: shape (..., 3, 3)
    Output:
        mrp: shape (..., 3)
    """

    b = _DCM_to_EulerParameters(dcm)  # (..., 4)

    mrp_0 = b[..., 1] / (1 + b[..., 0])
    mrp_1 = b[..., 2] / (1 + b[..., 0])
    mrp_2 = b[..., 3] / (1 + b[..., 0])
    mrp = torch.stack([mrp_0, mrp_1, mrp_2], dim=-1)
    return mrp


def _DCM_to_EulerParameters(C: torch.Tensor) -> torch.Tensor:
    """
    Convert a direction cosine matrix (DCM) to Euler Parameters (quaternion) with batched support.
    Ensures the first component is non-negative.
    Input:
        C: shape (..., 3, 3)
    Output:
        b: shape (..., 4)
    """
    tr = torch.einsum('... i i -> ...', C)

    b2_0 = (1 + tr) / 4.
    b2_1 = (1 + 2 * C[..., 0, 0] - tr) / 4.
    b2_2 = (1 + 2 * C[..., 1, 1] - tr) / 4.
    b2_3 = (1 + 2 * C[..., 2, 2] - tr) / 4.
    b2 = torch.stack([b2_0, b2_1, b2_2, b2_3], dim=-1)

    # Find the index of the maximum component
    i = torch.argmax(b2, dim=-1, keepdim=True)

    b = torch.zeros_like(b2)
    # case 0
    case0_mask = i == 0
    if torch.any(case0_mask):
        b_0 = torch.sqrt(b2_0)
        b_1 = (C[..., 1, 2] - C[..., 2, 1]) / (4 * b_0)
        b_2 = (C[..., 2, 0] - C[..., 0, 2]) / (4 * b_0)
        b_3 = (C[..., 0, 1] - C[..., 1, 0]) / (4 * b_0)
        b_case0 = torch.stack([b_0, b_1, b_2, b_3], dim=-1)
        b = torch.where(case0_mask, b_case0, b)

    # case 1
    case1_mask = i == 1
    if torch.any(case1_mask):
        b_1 = torch.sqrt(b2_1)
        b_0 = (C[..., 1, 2] - C[..., 2, 1]) / (4 * b_1)
        b_2 = (C[..., 0, 1] + C[..., 1, 0]) / (4 * b_1)
        b_3 = (C[..., 2, 0] + C[..., 0, 2]) / (4 * b_1)
        b_case1 = torch.stack([b_0, b_1, b_2, b_3], dim=-1)
        b = torch.where(case1_mask, b_case1, b)

    # case 2
    case2_mask = i == 2
    if torch.any(case2_mask):
        b_2 = torch.sqrt(b2_2)
        b_0 = (C[..., 2, 0] - C[..., 0, 2]) / (4 * b_2)
        b_0_negative_mask = b_0 < 0
        b_0 = torch.where(b_0_negative_mask, -b_0, b_0)
        b_2 = torch.where(b_0_negative_mask, -b_2, b_2)
        b_1 = (C[..., 0, 1] + C[..., 1, 0]) / (4 * b_2)
        b_3 = (C[..., 1, 2] + C[..., 2, 1]) / (4 * b_2)
        b_case2 = torch.stack([b_0, b_1, b_2, b_3], dim=-1)
        b = torch.where(case2_mask, b_case2, b)

    # case 3
    case3_mask = i == 3
    if torch.any(case3_mask):
        b_3 = torch.sqrt(b2_3)
        b_0 = (C[..., 0, 1] - C[..., 1, 0]) / (4 * b_3)
        b_0_negative_mask = b_0 < 0
        b_0 = torch.where(b_0_negative_mask, -b_0, b_0)
        b_3 = torch.where(b_0_negative_mask, -b_3, b_3)
        b_1 = (C[..., 2, 0] + C[..., 0, 2]) / (4 * b_3)
        b_2 = (C[..., 1, 2] + C[..., 2, 1]) / (4 * b_3)
        b_case3 = torch.stack([b_0, b_1, b_2, b_3], dim=-1)
        b = torch.where(case3_mask, b_case3, b)

    if not torch.all(case0_mask | case1_mask | case2_mask | case3_mask):
        raise RuntimeError("Invalid case")

    return b


def dcm_to_mrp(C: torch.Tensor) -> torch.Tensor:
    """
    将方向余弦矩阵(DCM)转换为修正罗德里格斯参数(MRP)
    
    参数:
        C: 方向余弦矩阵，形状为(*, 3, 3)，其中*表示任意数量的批次维度
        
    返回:
        sigma: 修正罗德里格斯参数，形状为(*, 3)
    """
    # 确保输入是3x3矩阵且为正交矩阵（这里仅做形状检查）
    assert C.shape[-2:] == (3, 3), "输入必须是形状为(*, 3, 3)的张量"

    # 计算矩阵的迹
    tr_C = torch.einsum('... i i -> ...', C)

    # 计算反对称分量向量e
    e1 = C[..., 2, 1] - C[..., 1, 2]  # c32 - c23
    e2 = C[..., 0, 2] - C[..., 2, 0]  # c13 - c31
    e3 = C[..., 1, 0] - C[..., 0, 1]  # c21 - c12
    e = torch.stack([e1, e2, e3], dim=-1)  # 形状为(*, 3)

    # 计算e的模平方
    e_squared = torch.sum(e**2, dim=-1, keepdim=True)  # 形状为(*, 1)

    # 处理特殊情况：当e接近零时（旋转角为0）
    # 使用小的epsilon来避免数值问题
    epsilon = 1e-8
    mask = e_squared < epsilon  # 形状为(*, 1)

    # 计算公式中的分子部分
    term = 1 + tr_C[..., None]  # 形状为(*, 1)
    sqrt_term = torch.sqrt(term**2 + e_squared)  # 形状为(*, 1)
    numerator = -term + sqrt_term  # 形状为(*, 1)

    # 计算MRP
    sigma = (numerator / e_squared) * e  # 形状为(*, 3)

    # 对于接近单位矩阵的情况，直接返回零向量
    sigma = torch.where(mask, torch.zeros_like(sigma), sigma)

    return -sigma


try:

    a = mrp_to_rotation_matrix(torch.rand(200, 3))
    b = dcm_to_mrp(a)

    c = _DCM_to_MRP(a)
    dcm_b = mrp_to_rotation_matrix(b)
    dcm_c = mrp_to_rotation_matrix(c)

    assert torch.allclose(
        dcm_b,
        dcm_c,
        atol=1e-5,
    )
    print("Success")
except AssertionError:
    print(dcm_b - dcm_c)
    print(b, c)
