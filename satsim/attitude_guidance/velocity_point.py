__all__ = ["VelocityPoint", "VelocityPointStateDict"]

from typing import TypedDict

import torch

from satsim.architecture import Module


class VelocityPointStateDict(TypedDict):
    pass


class VelocityPoint(Module[VelocityPointStateDict]):

    def __init__(self, mu: torch.Tensor | None = None, *args, **kwargs):
        '''This module generate the attitude reference to perform a constant pointing towards a Velocity orbit axis.

        Args:
            mu (torch.Tensor): The gravitational parameter of the central body.

        '''
        self.register_buffer(
            "_mu",
            mu,
            persistent=False,
        )

    @property
    def mu(self) -> torch.Tensor:
        return self.get_buffer('_mu')

    def forward(
        self,
        r_BN_N: torch.Tensor,
        v_BN_N: torch.Tensor,
        state_dict: VelocityPointStateDict | None = None,
        r_celestialObjectN_N: torch.Tensor | None = None,
        v_celestialObjectN_N: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> tuple[VelocityPointStateDict, tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor]]:
        '''This is the main method that gets called every time the module is updated.
            Args:
                state_dict (VelocityPointStateDict | None): The state dictionary of the module.
                r_BN_N: torch.Tensor | None = None: The position vector of the spacecraft in the inertial frame.
                v_BN_N: torch.Tensor | None = None: The velocity vector of the spacecraft in the inertial frame.
                r_celestialObjectN_N: torch.Tensor | None = None: The position vector of the celestial object in the inertial frame.
                v_celestialObjectN_N: torch.Tensor | None = None: The velocity vector of the celestial object in the inertial frame.
        '''
        # If the position vector of the celestial object is not provided, set it to zero
        if r_celestialObjectN_N is None:
            r_celestialObjectN_N = torch.tensor([0., 0., 0.],
                                                device=r_BN_N.device)
        # If the velocity vector of the celestial object is not provided, set it to zero
        if v_celestialObjectN_N is None:
            v_celestialObjectN_N = torch.tensor([0., 0., 0.],
                                                device=v_BN_N.device)

        # Calculate the relative position and velocity vectors
        rel_position_vector = r_BN_N - r_celestialObjectN_N
        rel_velocity_vector = v_BN_N - v_celestialObjectN_N

        rel_position_vector_magnitude = torch.norm(rel_position_vector, dim=-1)
        rel_velocity_vector_magnitude = torch.norm(rel_velocity_vector, dim=-1)

        dcm_RN_1 = rel_velocity_vector / rel_velocity_vector_magnitude.unsqueeze(
            -1)

        h = torch.cross(rel_position_vector, rel_velocity_vector, dim=-1)
        h_magnitude = torch.norm(h, dim=-1)
        dcm_RN_2 = h / h_magnitude.unsqueeze(-1)
        dcm_RN_0 = torch.cross(dcm_RN_1, dcm_RN_2, dim=-1)

        dcm_RN = torch.stack([dcm_RN_0, dcm_RN_1, dcm_RN_2], dim=-2)

        sigma_RN = _DCM_to_MRP(dcm_RN)

        omega_RN_R = torch.zeros_like(sigma_RN)
        dot_omega_RN_R = torch.zeros_like(sigma_RN)

        #### Robustness check ####
        mask = (rel_position_vector_magnitude > 1.0)  # shape: [batch]
        mask_expanded = mask.unsqueeze(-1)  # shape: [batch, 1]

        # Get the shape of the relative position vector
        ref_shape = rel_position_vector.shape
        batch_shape = ref_shape[:-1]

        # === Branch 1: r > 1.0 ===
        (oe_e, oe_f) = rv2elem(self.mu, rel_position_vector,
                               rel_velocity_vector)

        dot_f_dot_t_1 = h_magnitude / (rel_position_vector_magnitude**2)
        ddot_f_dot_t2_1 = -2.0 * (rel_velocity_vector * dcm_RN_0.sum(dim=-1).unsqueeze(-1)) \
                            / (rel_position_vector_magnitude ** 2).unsqueeze(-1) * dot_f_dot_t_1.unsqueeze(-1)

        denom = 1 + oe_e**2 + 2 * oe_e * torch.cos(oe_f)
        temp = (1 + oe_e * torch.cos(oe_f)) / denom

        omega_RN_R_2 = dot_f_dot_t_1 * temp  # shape: [batch]
        omega_RN_R_2 = omega_RN_R_2.unsqueeze(-1)  # shape: [batch, 1]

        omega_RN_R = torch.stack(
            [
                torch.zeros_like(omega_RN_R_2),  # omega_RN_R_0
                torch.zeros_like(omega_RN_R_2),  # omega_RN_R_1
                omega_RN_R_2  # omega_RN_R_2
            ],
            dim=-1)

        term2 = ddot_f_dot_t2_1.squeeze(-1) * temp \
                - dot_f_dot_t_1 ** 2 * oe_e * (oe_e**2 - 1) * torch.sin(oe_f) / (denom ** 2)
        dot_omega_RN_R_2 = term2.unsqueeze(-1)
        dot_omega_RN_R = torch.stack([
            torch.zeros_like(dot_omega_RN_R_2),
            torch.zeros_like(dot_omega_RN_R_2), dot_omega_RN_R_2
        ],
                                     dim=-1)  # shape: [batch, 3]

        # === Branch 2: r <= 1.0 ===
        zeros_scalar = torch.zeros(batch_shape + (1, ),
                                   device=rel_position_vector.device,
                                   dtype=rel_position_vector.dtype)
        zeros_vec = torch.zeros(batch_shape + (3, ),
                                device=rel_position_vector.device,
                                dtype=rel_position_vector.dtype)

        dot_f_dot_t_2 = zeros_scalar
        ddot_f_dot_t2_2 = zeros_scalar
        omega_RN_R_2 = zeros_vec
        dot_omega_RN_R_2 = zeros_vec

        # === Use torch.where to combine results ===
        dot_f_dot_t = torch.where(mask_expanded, dot_f_dot_t_1.unsqueeze(-1),
                                  dot_f_dot_t_2)
        ddot_f_dot_t2 = torch.where(mask_expanded, ddot_f_dot_t2_1,
                                    ddot_f_dot_t2_2)
        omega_RN_R = torch.where(mask_expanded.expand_as(omega_RN_R),
                                 omega_RN_R, zeros_vec)
        dot_omega_RN_R = torch.where(mask_expanded.expand_as(dot_omega_RN_R),
                                     dot_omega_RN_R, zeros_vec)

        dcm_NR = dcm_RN.transpose(-1, -2)
        omega_RN_N = torch.matmul(dcm_NR, omega_RN_R.unsqueeze(-1)).squeeze(-1)
        dot_omega_RN_N = torch.matmul(dcm_NR,
                                      dot_omega_RN_R.unsqueeze(-1)).squeeze(-1)

        return VelocityPointStateDict(), (sigma_RN, omega_RN_N, dot_omega_RN_N)


def _DCM_to_MRP(dcm: torch.Tensor) -> torch.Tensor:
    """
    Convert a direction cosine matrix to a MRP (Modified Rodrigues Parameters) vector.
    Supports batch input.

    Input:
    dcm: tensor with shape (..., 3, 3)

    Output:
    mrp: tensor with shape (..., 3)
    """

    b = _DCM_to_EulerParameters(dcm)

    mrp_0 = b[..., 1] / (1 + b[..., 0])
    mrp_1 = b[..., 2] / (1 + b[..., 0])
    mrp_2 = b[..., 3] / (1 + b[..., 0])
    mrp = torch.stack([mrp_0, mrp_1, mrp_2], dim=-1)
    return mrp


def _DCM_to_EulerParameters(C: torch.Tensor) -> torch.Tensor:
    """
    Convert a direction cosine matrix (DCM) to Euler Parameters (quaternion) with batched support.
    Ensures the first component is non-negative
    Input:
        C: shape (..., 3, 3)
    Output:
        b: shape (..., 4)
    """
    tr = C[..., 0, 0] + C[..., 1, 1] + C[..., 2, 2]

    b2_0 = (1 + tr) / 4.
    b2_1 = (1 + 2 * C[..., 0, 0] - tr) / 4.
    b2_2 = (1 + 2 * C[..., 1, 1] - tr) / 4.
    b2_3 = (1 + 2 * C[..., 2, 2] - tr) / 4.
    b2 = torch.stack([b2_0, b2_1, b2_2, b2_3], dim=-1)

    # Find the index of the maximum component
    i = torch.argmax(b2, dim=-1)
    #b = torch.empty(C.shape[:-2] + (4,), dtype=C.dtype, device=C.device)

    # case 0
    if i == 0:
        # 分别计算b_0, b_1, b_2, b_3，然后stack
        b_0 = torch.sqrt(b2[..., 0])
        b_1 = (C[..., 1, 2] - C[..., 2, 1]) / (4 * b_0)
        b_2 = (C[..., 2, 0] - C[..., 0, 2]) / (4 * b_0)
        b_3 = (C[..., 0, 1] - C[..., 1, 0]) / (4 * b_0)
        b = torch.stack([b_0, b_1, b_2, b_3], dim=-1)
        return b

    # case 1
    elif i == 1:
        b_1 = torch.sqrt(b2[..., 1])
        b_0 = (C[..., 1, 2] - C[..., 2, 1]) / (4 * b_1)
        b_2 = (C[..., 0, 1] + C[..., 1, 0]) / (4 * b_1)
        b_3 = (C[..., 2, 0] + C[..., 0, 2]) / (4 * b_1)
        b = torch.stack([b_0, b_1, b_2, b_3], dim=-1)
        return b

    # case 2
    elif i == 2:
        b_2 = torch.sqrt(b2[..., 2])
        b_0 = (C[..., 2, 0] - C[..., 0, 2]) / (4 * b_2)
        if b_0 < 0:
            b_0 = -b_0
            b_2 = -b_2
        b_1 = (C[..., 0, 1] + C[..., 1, 0]) / (4 * b_2)
        b_3 = (C[..., 1, 2] + C[..., 2, 1]) / (4 * b_2)
        b = torch.stack([b_0, b_1, b_2, b_3], dim=-1)
        return b

    # case 3
    elif i == 3:
        b_3 = torch.sqrt(b2[..., 3])
        b_0 = (C[..., 0, 1] - C[..., 1, 0]) / (4 * b_3)
        if b_0 < 0:
            b_0 = -b_0
            b_3 = -b_3
        b_1 = (C[..., 2, 0] + C[..., 0, 2]) / (4 * b_3)
        b_2 = (C[..., 1, 2] + C[..., 2, 1]) / (4 * b_3)
        b = torch.stack([b_0, b_1, b_2, b_3], dim=-1)
        return b
    else:
        raise ValueError(
            "Invalid index for DCM to Euler Parameters conversion.")


def rv2elem(mu: torch.Tensor, r_vector: torch.Tensor,
            v_vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    
    Purpose: Translates the orbit elements inertial Cartesian position
    vector rVec and velocity vector vVec into the corresponding
    classical orbit elements where
            a   - semi-major axis (zero if parabolic)
            e   - eccentricity
            i   - inclination               (rad)
            AN  - ascending node            (rad)
            AP  - argument of periapses     (rad)
            f   - true anomaly angle        (rad)
                    if the orbit is rectilinear, then this will be the
                    eccentric or hyperbolic anomaly
            rp  - radius at periapses
            ra  - radius at apoapses (zero if parabolic)
    The attracting body is specified through the supplied
    gravitational constant mu (units of km^3/s^2).
    Inputs:
    mu = gravitational parameter
    rVec = position vector
    vVec = velocity vector
    Outputs:
    elements = orbital elements
    
    
    """
    eps = 1e-12

    # Create unit vectors that match the batch dimensions of input tensors
    # Get the shape of r_vector excluding the last dimension (which should be 3 for x,y,z)
    batch_shape = r_vector.shape[:-1]
    n1_hat = torch.tensor([1, 0, 0],
                          device=r_vector.device,
                          dtype=r_vector.dtype).expand(*batch_shape, 3)
    n3_hat = torch.tensor([0, 0, 1],
                          device=r_vector.device,
                          dtype=r_vector.dtype).expand(*batch_shape, 3)

    r_magnitude = torch.linalg.norm(r_vector, dim=-1)
    v_magnitude = torch.linalg.norm(v_vector, dim=-1)

    ir_hat = r_vector / r_magnitude.unsqueeze(-1)

    h_vector = torch.cross(r_vector, v_vector, dim=-1)
    h_magnitude = torch.linalg.norm(h_vector, dim=-1)
    ih_hat = h_vector / h_magnitude.unsqueeze(-1)
    p = h_magnitude**2 / mu

    #Calculate the line of nodes
    n_vector = torch.cross(n3_hat, h_vector, dim=-1)
    n_magnitude = torch.linalg.norm(n_vector, dim=-1)
    if n_magnitude < eps:
        in_hat = n1_hat
    else:
        in_hat = n_vector / n_magnitude.unsqueeze(-1)

    #Orbit eccentricity vector
    e_vector = (v_magnitude**2 / mu - 1.0 / r_magnitude) * r_vector
    v_3 = ((r_vector * v_vector).sum(dim=-1) / mu) * v_vector
    e_vector = e_vector - v_3
    e_magnitude = torch.linalg.norm(e_vector, dim=-1)
    r_periap = p / (1 + e_magnitude)

    #Orbit eccentricity units vector
    if e_magnitude > eps:
        ie_hat = e_vector / e_magnitude.unsqueeze(-1)
    else:
        ie_hat = in_hat

    #compute semi-major axis
    alpha = 2.0 / r_magnitude - v_magnitude**2 / mu
    if alpha > eps:
        #elliptic or hyperbolic case
        a = 1.0 / alpha
        r_apoap = p / (1 - e_magnitude)
    else:
        #parabolic case
        a = torch.zeros([0], device=r_vector.device, dtype=r_vector.dtype)
        r_apoap = torch.zeros([0],
                              device=r_vector.device,
                              dtype=r_vector.dtype)

    # Calculate inclination
    i = torch.acos(torch.clamp(h_vector[..., 2] / h_magnitude, -1, 1))

    # Calculate Ascending Node Omega
    v_3 = torch.cross(n1_hat, ih_hat, dim=-1)
    Omega = torch.atan2(v_3[..., 2], in_hat[..., 0])

    if Omega < 0:
        Omega += 2 * torch.pi

    # Calculate argument of periapsis omega
    v_3 = torch.cross(in_hat, ie_hat, dim=-1)
    omega = torch.atan2((ih_hat * v_3).sum(dim=-1),
                        (in_hat * ie_hat).sum(dim=-1))

    if omega < 0:
        omega += 2 * torch.pi

    # Calculate true anomaly angle f
    v_3 = torch.cross(ie_hat, ir_hat, dim=-1)
    f_atan_1 = (ih_hat * v_3).sum(dim=-1)
    f_atan_2 = (ie_hat * ir_hat).sum(dim=-1)
    f = torch.atan2(f_atan_1, f_atan_2)

    if f < 0:
        f += 2 * torch.pi

    return (e_magnitude, f)
