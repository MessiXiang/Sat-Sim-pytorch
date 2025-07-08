__all__ = ['create_skew_symmetric_matrix', 'Bmat', 'to_rotation_matrix']
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
    if vector.shape != (3, ):
        raise ValueError("Input vector must be a 3D tensor")

    v1, v2, v3 = vector.unbind()

    row0 = torch.stack([torch.zeros_like(v1), -v3, v2])
    row1 = torch.stack([v3, torch.zeros_like(v1), -v1])
    row2 = torch.stack([-v2, v1, torch.zeros_like(v1)])

    return torch.stack([row0, row1, row2])


def Bmat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the 3x3 B matrix for a single 3D vector using a simplified form.
    Input:
        v: torch.Tensor of shape [3], representing the input vector [x, y, z]
    Output:
        B: torch.Tensor of shape [3, 3], the computed B matrix
    """
    # Validate input
    if v.shape != (3, ):
        raise ValueError("Input must be a 1D tensor of shape [3]")

    # Compute 1 - ||v||^2
    ms2 = 1.0 - torch.sum(v**2)

    # Compute outer product vv^T
    vvT = torch.outer(v, v)  # v @ v.T

    # Compute skew-symmetric matrix [v]_\times
    S = create_skew_symmetric_matrix(v)

    # Compute B = (1 - ||v||^2)I + 2vv^T + 2[v]_\times
    I = torch.eye(3, device=v.device)
    B = ms2 * I + 2 * vvT + 2 * S

    return B


def to_rotation_matrix(vec: torch.Tensor) -> torch.Tensor:
    """
    Converts a 3D vector to a 3x3 rotation matrix using optimized PyTorch operations.

    This function computes a 3x3 rotation matrix from a 3D vector based on a specific
    mathematical formulation involving the vector's squared norm, cross terms, and
    normalization. The resulting matrix is derived from the vector's components (x, y, z)
    and is normalized by the square of (1 + ||v||^2).

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
    if vec.shape != (3, ):
        raise ValueError(
            f"Expected input tensor of shape [3], got {vec.shape}")

    # Compute squared norm and related terms
    vec_sq = vec**2
    norm_sq = vec_sq.sum()  # ||v||^2
    ps2 = 1.0 + norm_sq  # 1 + ||v||^2
    ms2 = 1.0 - norm_sq  # 1 - ||v||^2
    ms2_sq = ms2**2  # (1 - ||v||^2)^2

    # Compute squared terms (x^2, y^2, z^2)
    # [x^2, y^2, z^2]

    # Compute cross terms (8xy, 8xz, 8yz) efficiently
    cross_terms = 8.0 * vec.outer(
        vec)  # Outer product: [x, y, z] * [x, y, z]^T
    s1s2, s1s3, s2s3 = cross_terms[0, 1], cross_terms[0, 2], cross_terms[1, 2]

    # Diagonal terms: 4 * (s1_sq - s2_sq - s3_sq) + ms2_sq, etc.
    diag_coeffs = 4.0 * torch.stack([
        vec_sq[0] - vec_sq[1] - vec_sq[2],  # s1_sq - s2_sq - s3_sq
        -vec_sq[0] + vec_sq[1] - vec_sq[2],  # -s1_sq + s2_sq - s3_sq
        -vec_sq[0] - vec_sq[1] + vec_sq[2]  # -s1_sq - s2_sq + s3_sq
    ])
    diag_coeffs = diag_coeffs + ms2_sq

    # Off-diagonal terms

    m01 = s1s2 - 4.0 * vec[2] * ms2
    m02 = s1s3 + 4.0 * vec[1] * ms2
    m10 = s1s2 + 4.0 * vec[2] * ms2
    m12 = s2s3 - 4.0 * vec[0] * ms2
    m20 = s1s3 - 4.0 * vec[1] * ms2
    m21 = s2s3 + 4.0 * vec[0] * ms2

    res = torch.stack([
        torch.stack([diag_coeffs[0], m01, m02]),
        torch.stack([m10, diag_coeffs[1], m12]),
        torch.stack([m20, m21, diag_coeffs[2]])
    ])

    # Normalize by (ps2)^2
    res = res / (ps2**2)

    return res
