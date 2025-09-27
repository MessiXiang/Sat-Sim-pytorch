__all__ = ["HillPoint", "HillPointStateDict"]

from typing import TypedDict
import torch
from satsim.architecture import Module
from satsim.utils import dcm_to_mrp
from satsim.simulation.gravity import Ephemeris


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
        ephemeris: Ephemeris,
        state_dict: HillPointStateDict | None = None,
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

        r_PN_N = ephemeris['position_in_inertial']

        v_PN_N = ephemeris['velocity_in_inertial']

        r_BP_N = r_BN_N - r_PN_N
        v_BP_N = v_BN_N - v_PN_N

        rhat_BP_N = r_BP_N / torch.norm(r_BP_N, dim=-1, keepdim=True)

        h = torch.cross(r_BP_N, v_BP_N, dim=-1)
        h_magnitude = torch.norm(h, dim=-1)
        orbit_normal_vector = h / h_magnitude.unsqueeze(-1)
        reference_base_vector = torch.cross(
            orbit_normal_vector,
            rhat_BP_N,
            dim=-1,
        )

        # Here define a reference frame where x // PB, z // orbit_normal_vector
        # y is defined by right-hand rule
        dcm_RN = torch.stack(
            [rhat_BP_N, reference_base_vector, orbit_normal_vector], dim=-2)

        sigma_RN = dcm_to_mrp(dcm_RN)

        BP_distance = torch.norm(r_BP_N, dim=-1)

        # find the shape of the rel_position_vector
        ref_shape = r_BP_N.shape
        batch_shape = ref_shape[:-1]

        # Create a mask to identify elements where the magnitude of the relative position vector is greater than 1.0
        mask = BP_distance > 1.0

        # Calculate the value of dot_f_dot_t and ddot_f_dot_t2 for all elements (without considering the if)
        dot_f_dot_t_all = h_magnitude / (BP_distance**2)

        # Broadcast processing of rel_velocity_vector * dcm_RN_0
        rel_velocity_proj = (v_BP_N * rhat_BP_N).sum(dim=-1)
        ddot_f_dot_t2_all = -2.0 * rel_velocity_proj / BP_distance * dot_f_dot_t_all

        zeros = torch.zeros(batch_shape + (1, ),
                            device=r_BP_N.device,
                            dtype=r_BP_N.dtype)

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
