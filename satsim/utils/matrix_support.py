__all__ = [
    'create_skew_symmetric_matrix',
    'Bmat',
    'to_rotation_matrix',
    'add_mrp',
    'sub_mrp',
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
    term1 = ms2 * torch.eye(3, device=ms2.device)
    term2 = 2 * create_skew_symmetric_matrix(mrp)
    term3 = 2 * torch.einsum('... i, ... j -> ... i j', mrp, mrp)

    return term1 + term2 + term3


def to_rotation_matrix(mrp: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of Modified Rodrigues Parameters (MRP) to rotation matrices.
    
    Args:
        mrp: torch.Tensor of shape [batch_size, 3], where each row is an MRP vector (x, y, z).
             Assumed to be float32 with requires_grad=True for gradient tracking.
    
    Returns:
        torch.Tensor of shape [batch_size, 3, 3], where each [3, 3] slice is a rotation matrix.
    """

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
    r01 = s1s2 - 4.0 * z * ms2.squeeze()  # [batch_size]
    r02 = s1s3 + 4.0 * y * ms2.squeeze()  # [batch_size]
    r10 = s1s2 + 4.0 * z * ms2.squeeze()  # [batch_size]
    r11 = 4.0 * (-s1_sq + s2_sq - s3_sq) + ms2_sq.squeeze()  # [batch_size]
    r12 = s2s3 - 4.0 * x * ms2.squeeze()  # [batch_size]
    r20 = s1s3 - 4.0 * y * ms2.squeeze()  # [batch_size]
    r21 = s2s3 + 4.0 * x * ms2.squeeze()  # [batch_size]
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

    cross = torch.cross(q1, q2)
    term1 = (1 - (dot2)) * q1
    term2 = 2 * cross
    term3 = (1 - (dot1)) * q2

    result = term1 + term2 - term3
    result = result / den

    norm2 = torch.sum(result * result, dim=-1, keepdim=True)
    result = torch.where(norm2 > 1.0, -result / norm2, result)

    return result
