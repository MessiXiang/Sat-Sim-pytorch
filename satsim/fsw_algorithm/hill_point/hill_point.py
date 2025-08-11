__all__ = ["HillPoint", "HillPointStateDict"]

from typing import TypedDict
import torch
from satsim.architecture import Module


class HillPointStateDict(TypedDict):
    pass


class HillPoint(Module[HillPointStateDict]):

    def __init__(self, *args, **kwargs):
        '''This module generate the attitude reference to perform a constant pointing towards a Hill frame orbit axis
        '''
        super().__init__(*args, **kwargs)

    def forward(
        self,
        r_BN_N: torch.Tensor,
        v_BN_N: torch.Tensor,
        state_dict: HillPointStateDict | None = None,
        r_celestialObjectN_N: torch.Tensor | None = None,
        v_celestialObjectN_N: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> tuple[HillPointStateDict, tuple[torch.Tensor, torch.Tensor,
                                         torch.Tensor]]:
        '''This is the main method that gets called every time the module is updated.
            Args:
                state_dict (HillPointStateDict | None): The state dictionary of the module.
                r_BN_N: torch.Tensor: The position vector of the spacecraft in the inertial frame.
                v_BN_N: torch.Tensor: The velocity vector of the spacecraft in the inertial frame.
                r_celestialObjectN_N: torch.Tensor | None = None: The position vector of the celestial object in the inertial frame.
                v_celestialObjectN_N: torch.Tensor | None = None: The velocity vector of the celestial object in the inertial frame.
        '''

        if r_celestialObjectN_N is None:
            r_celestialObjectN_N = torch.tensor([0., 0., 0.],
                                                device=r_BN_N.device)
        if v_celestialObjectN_N is None:
            v_celestialObjectN_N = torch.tensor([0., 0., 0.],
                                                device=v_BN_N.device)

        rel_position_vector = r_BN_N - r_celestialObjectN_N
        rel_velocity_vector = v_BN_N - v_celestialObjectN_N

        rel_position_vector_magnitude = torch.norm(rel_position_vector, dim=-1)

        dcm_RN_0 = rel_position_vector / rel_position_vector_magnitude.unsqueeze(
            -1)

        h = torch.cross(rel_position_vector, rel_velocity_vector, dim=-1)
        h_magnitude = torch.norm(h, dim=-1)
        dcm_RN_2 = h / h_magnitude.unsqueeze(-1)
        dcm_RN_1 = torch.cross(dcm_RN_2, dcm_RN_0, dim=-1)

        dcm_RN = torch.stack([dcm_RN_0, dcm_RN_1, dcm_RN_2], dim=-2)

        sigma_RN = _DCM_to_MRP(dcm_RN)

        r_magnitude = torch.norm(rel_position_vector, dim=-1)

        # find the shape of the rel_position_vector
        ref_shape = rel_position_vector.shape
        batch_shape = ref_shape[:-1]

        # Create a mask to identify elements where the magnitude of the relative position vector is greater than 1.0
        mask = r_magnitude > 1.0

        # Calculate the value of dot_f_dot_t and ddot_f_dot_t2 for all elements (without considering the if)
        dot_f_dot_t_all = h_magnitude / (r_magnitude**2)

        # Broadcast processing of rel_velocity_vector * dcm_RN_0
        rel_velocity_proj = (rel_velocity_vector * dcm_RN_0).sum(dim=-1)
        ddot_f_dot_t2_all = -2.0 * rel_velocity_proj / r_magnitude * dot_f_dot_t_all

        zeros = torch.zeros(batch_shape + (1, ),
                            device=rel_position_vector.device,
                            dtype=rel_position_vector.dtype)

        # Use torch.where to select results element by element (and keep the last dimension)
        dot_f_dot_t = torch.where(mask.unsqueeze(-1),
                                  dot_f_dot_t_all.unsqueeze(-1), zeros)
        ddot_f_dot_t2 = torch.where(mask.unsqueeze(-1),
                                    ddot_f_dot_t2_all.unsqueeze(-1), zeros)

        omega_RN_R_2 = dot_f_dot_t
        omega_RN_R_0 = torch.zeros_like(dot_f_dot_t)
        omega_RN_R_1 = torch.zeros_like(dot_f_dot_t)
        omega_RN_R = torch.stack([omega_RN_R_0, omega_RN_R_1, omega_RN_R_2],
                                 dim=-1)

        omega_dot_RN_R_2 = ddot_f_dot_t2
        omega_dot_RN_R_0 = torch.zeros_like(ddot_f_dot_t2)
        omega_dot_RN_R_1 = torch.zeros_like(ddot_f_dot_t2)
        omega_dot_RN_R = torch.stack(
            [omega_dot_RN_R_0, omega_dot_RN_R_1, omega_dot_RN_R_2], dim=-1)

        dcm_NR = dcm_RN.transpose(-1, -2)

        omega_RN_N = torch.matmul(dcm_NR, omega_RN_R.unsqueeze(-1)).squeeze(-1)
        dot_omega_RN_N = torch.matmul(dcm_NR,
                                      omega_dot_RN_R.unsqueeze(-1)).squeeze(-1)

        return HillPointStateDict(), (sigma_RN, omega_RN_N, dot_omega_RN_N)


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
    tr = C[..., 0, 0] + C[..., 1, 1] + C[..., 2, 2]

    b2_0 = (1 + tr) / 4.
    b2_1 = (1 + 2 * C[..., 0, 0] - tr) / 4.
    b2_2 = (1 + 2 * C[..., 1, 1] - tr) / 4.
    b2_3 = (1 + 2 * C[..., 2, 2] - tr) / 4.
    b2 = torch.stack([b2_0, b2_1, b2_2, b2_3], dim=-1)

    # Find the index of the maximum component
    i = torch.argmax(b2, dim=-1)

    # case 0
    if i == 0:
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
        raise ValueError("Invalid index for case")
