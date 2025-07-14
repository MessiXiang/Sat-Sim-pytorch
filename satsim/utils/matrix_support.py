__all__ = [
    'create_skew_symmetric_matrix',
    'Bmat',
    'to_rotation_matrix',
    'addMRP',
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


def Bmat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the 3x3 B matrix for a single 3D vector using a simplified form.
    Input:
        v: torch.Tensor of shape [3], representing the input vector [x, y, z]
    Output:
        B: torch.Tensor of shape [3, 3], the computed B matrix
    """
    # Validate input
    if v.shape[-1] != 3:
        raise ValueError(f"Input last dim must be of shape [3], got {v.shape}")

    # Compute 1 - ||v||^2
    ms2 = 1.0 - torch.sum(v**2, dim=-1, keepdim=True).unsqueeze(-1)

    # Compute outer product vv^T
    vvT = torch.matmul(v.unsqueeze(-1), v.unsqueeze(-2))  # v @ v.T

    # Compute skew-symmetric matrix [v]_\times
    S = create_skew_symmetric_matrix(v)

    # Compute B = (1 - ||v||^2)I + 2vv^T + 2[v]_\times
    I = torch.eye(3, device=v.device)
    B = ms2 * I + 2 * vvT + 2 * S

    return B


def to_rotation_matrix(vec: torch.Tensor) -> torch.Tensor:
    """
    This function computes a 3x3 rotation matrix from a MRP vector.

    Args:
        vec (torch.Tensor): Input tensor of shape [3] representing a 3D vector (x, y, z).

    Returns:
        torch.Tensor: Output tensor of shape [3, 3] representing the rotation matrix.

    Raises:
        ValueError: If the input tensor does not have shape [3].

    Notes:
        The computation follows the formula from the original C++ implementation:
        - Diagonal terms: 4 * (x^2 - y^2 - z^2) + (1 - ||v||^2)^2, etc.
        - Off-diagonal terms: 8xy ± 4z(1 - ||v||^2), etc.
        - Normalization: Divide by (1 + ||v||^2)^2.
        The implementation uses vectorized operations for efficiency and supports
        GPU acceleration by inheriting the input tensor's dtype and device.
    """
    q1, q2, q3 = vec.unbind(-1)
    q1_sq, q2_sq, q3_sq = q1**2, q2**2, q3**2
    d1 = q1_sq + q2_sq + q3_sq
    S = 1 - d1
    d = (1 + d1)**2

    c00 = 4 * (2 * q1_sq - d1) + S**2
    c01 = 8 * q1 * q2 + 4 * q3 * S
    c02 = 8 * q1 * q3 - 4 * q2 * S
    c10 = 8 * q2 * q1 - 4 * q3 * S
    c11 = 4 * (2 * q2_sq - d1) + S**2
    c12 = 8 * q2 * q3 + 4 * q1 * S
    c20 = 8 * q3 * q1 + 4 * q2 * S
    c21 = 8 * q3 * q2 - 4 * q1 * S
    c22 = 4 * (2 * q3_sq - d1) + S**2

    C = torch.stack(
        [
            torch.stack([c00, c01, c02], dim=-1),
            torch.stack([c10, c11, c12], dim=-1),
            torch.stack([c20, c21, c22], dim=-1),
        ],
        dim=-2,
    )
    return C / d.unsqueeze(-1).unsqueeze(-1)


def addMRP(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
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
