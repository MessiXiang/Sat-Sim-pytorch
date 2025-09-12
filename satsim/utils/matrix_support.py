__all__ = [
    'create_skew_symmetric_matrix',
    'Bmat',
    'mrp_to_rotation_matrix',
    'add_mrp',
    'sub_mrp',
    'dcm_to_mrp',
    'dcm_to_eulerparameters',
]
import torch


def create_skew_symmetric_matrix(vector: torch.Tensor) -> torch.Tensor:
    """
    Create a skew-symmetric matrix from a 3D vector for cross-product representation.

    This function constructs a 3x3 skew-symmetric matrix from a given 3D vector,
    which can be used to represent the cross product operation as a matrix multiplication.
    For a vector v = [v1, v2, v3], the skew-symmetric matrix is:
        [[ 0, -v3,  v2],
         [ v3,  0, -v1],
         [-v2,  v1,  0]]

    Args:
        vector (torch.Tensor): Input 3D vector (shape: [3]).

    Returns:
        torch.Tensor: Skew-symmetric matrix (shape: [3, 3]).

    Raises:
        ValueError: If the input vector is not a 3D tensor.
    """
    if vector.shape[-1] != 3:
        raise ValueError(
            f"Input last dim must be of shape [3], got {vector.shape}")

    v1, v2, v3 = vector.unbind(-1)

    row0 = torch.stack([torch.zeros_like(v1), -v3, v2], dim=-1)
    row1 = torch.stack([v3, torch.zeros_like(v1), -v1], dim=-1)
    row2 = torch.stack([-v2, v1, torch.zeros_like(v1)], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)


def Bmat(mrp: torch.Tensor) -> torch.Tensor:
    """
    Computes the B-matrix for a batch of Modified Rodrigues Parameters (MRP).
    
    Args:
        mrp: torch.Tensor of shape [batch_size, 3], where each row is an MRP vector (x, y, z).
             Assumed to be float32 with requires_grad=True for gradient tracking.
    
    Returns:
        torch.Tensor of shape [batch_size, 3, 3], where each [3, 3] slice is a B-matrix.
    """
    # Compute intermediate values
    ms2 = 1.0 - torch.sum(mrp**2, dim=-1)
    # TODO: change to einops and add to checklist
    term1 = ms2.unsqueeze(-1).unsqueeze(-1) * torch.eye(3, device=ms2.device)
    term2 = 2 * create_skew_symmetric_matrix(mrp)
    term3 = 2 * torch.einsum('... i, ... j -> ... i j', mrp, mrp)

    return term1 + term2 + term3


def mrp_to_rotation_matrix(mrp: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of Modified Rodrigues Parameters (MRP) attitude_BN to rotation matrices direction_cosine_matrix_BN.
    
    Args:
        mrp: torch.Tensor of shape [batch_size, 3], where each row is an MRP vector (x, y, z).
             Assumed to be float32 with requires_grad=True for gradient tracking.
    
    Returns:
        torch.Tensor of shape [batch_size, 3, 3], where each [3, 3] slice is a rotation matrix.
    """
    # NOTE: Originaly return dcm_NB, but here we return dcm_BN
    # Compute intermediate values
    ps2 = 1.0 + torch.sum(mrp**2, dim=-1, keepdim=True)  # [batch_size, 1]
    ms2 = 1.0 - torch.sum(mrp**2, dim=-1, keepdim=True)  # [batch_size, 1]
    ms2_sq = ms2 * ms2  # [batch_size, 1]

    x, y, z = mrp.unbind(-1)  # [batch_size]

    s1s2 = 8.0 * x * y  # [batch_size]
    s1s3 = 8.0 * x * z  # [batch_size]
    s2s3 = 8.0 * y * z  # [batch_size]

    s1_sq = x * x  # [batch_size]
    s2_sq = y * y  # [batch_size]
    s3_sq = z * z  # [batch_size]

    # Construct rotation matrix elements
    r00 = 4.0 * (s1_sq - s2_sq - s3_sq) + ms2_sq.squeeze()  # [batch_size]
    r10 = s1s2 - 4.0 * z * ms2.squeeze()  # [batch_size]
    r20 = s1s3 + 4.0 * y * ms2.squeeze()  # [batch_size]
    r01 = s1s2 + 4.0 * z * ms2.squeeze()  # [batch_size]
    r11 = 4.0 * (-s1_sq + s2_sq - s3_sq) + ms2_sq.squeeze()  # [batch_size]
    r21 = s2s3 - 4.0 * x * ms2.squeeze()  # [batch_size]
    r02 = s1s3 - 4.0 * y * ms2.squeeze()  # [batch_size]
    r12 = s2s3 + 4.0 * x * ms2.squeeze()  # [batch_size]
    r22 = 4.0 * (-s1_sq - s2_sq + s3_sq) + ms2_sq.squeeze()  # [batch_size]

    # Stack elements into [batch_size, 3, 3] tensor
    res = torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1)
        ],
        dim=-2,
    )  # [batch_size, 3, 3]

    # Normalize the rotation matrix
    res = res / (ps2 * ps2).unsqueeze(-1)  # Broadcasting to [batch_size, 3, 3]

    return res


def add_mrp(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    dot1 = torch.sum(q1 * q1, dim=-1, keepdim=True)  # ...,1
    dot2 = torch.sum(q2 * q2, dim=-1, keepdim=True)  # ...,1
    dot12 = torch.sum(q1 * q2, dim=-1, keepdim=True)  # ...,1

    den = 1 + dot1 * dot2 - 2 * dot12

    mask = torch.abs(den) < 0.1
    if mask.any():
        q2_new = -q2 / dot2
        q2 = torch.where(mask, q2_new, q2)

        dot2 = torch.sum(q2 * q2, dim=-1, keepdim=True)
        den = 1 + dot1 * dot2 - 2 * dot12

    term1 = (1 - dot1) * q2
    term2 = (1 - dot2) * q1
    cross = torch.cross(q1, q2, dim=-1)
    num = term1 + term2 + 2 * cross

    q = num / den

    q_norm_sq = torch.sum(q * q, dim=-1, keepdim=True)
    large_norm_mask = q_norm_sq > 1.0
    if large_norm_mask.any():
        q = torch.where(large_norm_mask, -q / q_norm_sq, q)

    return q


def sub_mrp(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    dot1 = torch.sum(q1 * q1, dim=-1, keepdim=True)  # ...,1
    dot2 = torch.sum(q2 * q2, dim=-1, keepdim=True)  # ...,1
    dot12 = torch.sum(q1 * q2, dim=-1, keepdim=True)  # ...,1
    den = 1 + dot1 * dot2 + 2 * dot12

    mask = torch.abs(den) < 0.1
    if mask.any():
        q1_new = -q1 / dot1
        q1 = torch.where(mask, q1_new, q2)
        den = 1 + dot1 * dot2 + 2 * dot12

    cross = torch.cross(q1, q2, dim=-1)
    term1 = (1 - (dot2)) * q1
    term2 = 2 * cross
    term3 = (1 - (dot1)) * q2

    result = term1 + term2 - term3
    result = result / den

    norm2 = torch.sum(result * result, dim=-1, keepdim=True)
    result = torch.where(norm2 > 1.0, -result / norm2, result)

    return result


def dcm_to_mrp(dcm: torch.Tensor) -> torch.Tensor:
    """
    Convert a direction cosine matrix to a MRP vector. 
    Input:
        dcm: shape (..., 3, 3)
    Output:
        mrp: shape (..., 3)

    Note:
    with torch.float32, this method has a accurate around 1e-6
    """

    b = dcm_to_eulerparameters(dcm)  # (..., 4)

    mrp_0 = b[..., 1] / (1 + b[..., 0])
    mrp_1 = b[..., 2] / (1 + b[..., 0])
    mrp_2 = b[..., 3] / (1 + b[..., 0])
    mrp = torch.stack([mrp_0, mrp_1, mrp_2], dim=-1)
    return mrp


def dcm_to_eulerparameters(C: torch.Tensor) -> torch.Tensor:
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
