__all__ = [
    'HubEffector',
    'HubEffectorDynamicParams',
    'HubEffectorStateDict',
]
from typing import TypedDict

import torch

from satsim.utils import Bmat, to_rotation_matrix

from ..base.state_effector import BackSubMatrices, MassProps, BaseStateEffector, StateEffectorStateDict


class HubEffectorDynamicParams(TypedDict):
    pos: torch.Tensor  # [3]
    velocity: torch.Tensor  # [3]
    sigma: torch.Tensor  # [3]
    omega: torch.Tensor  # [3]
    grav_velocity: torch.Tensor  # [3]
    grav_velocity_bc: torch.Tensor  # [3]


HubEffectorStateDict = StateEffectorStateDict[HubEffectorDynamicParams]


# MRPSwitchCount deserted
class HubEffector(
        BaseStateEffector[HubEffectorStateDict], ):

    def __init__(
        self,
        *args,
        mass: torch.Tensor,
        moment_of_inertia_matrix_wrt_body_point: torch.Tensor,
        pos: torch.Tensor,
        velocity: torch.Tensor,
        sigma: torch.Tensor | None = None,
        omega: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        mass = torch.tensor([1.]) if mass is None else mass
        moment_of_inertia_matrix_wrt_body_point = torch.eye(
            3, 3
        ) if moment_of_inertia_matrix_wrt_body_point is None else \
        moment_of_inertia_matrix_wrt_body_point

        self.register_buffer(
            'mass',
            mass,
            persistent=False,
        )
        self.register_buffer(
            'moment_of_inertia_matrix_wrt_body_point',
            moment_of_inertia_matrix_wrt_body_point,
            persistent=False,
        )

        self.pos_init = torch.zeros(3) if pos is None else pos
        self.velocity_init = torch.zeros(3) if velocity is None else velocity
        self.sigma_init = torch.zeros(3) if sigma is None else sigma
        self.omega_init = torch.zeros(3) if omega is None else omega

    def reset(self) -> HubEffectorStateDict:
        state_dict = super().reset()

        mass_props: MassProps = dict(
            mass=self.get_buffer('mass'),
            moment_of_inertia_matrix_wrt_body_point=self.get_buffer(
                'moment_of_inertia_matrix_wrt_body_point'),
        )

        dynamic_params = HubEffectorDynamicParams(
            pos=self.pos_init.clone(),
            velocity=self.velocity_init.clone(),
            sigma=self.sigma_init.clone(),
            omega=self.omega_init.clone(),
            grav_velocity=self.velocity_init.clone(),
            grav_velocity_bc=self.velocity_init.clone(),
        )
        state_dict.update(dynamic_params=dynamic_params, mass_props=mass_props)

        return state_dict

    def forward(
        self,
        state_dict: HubEffectorStateDict,
        *args,
        **kwargs,
    ) -> tuple[HubEffectorStateDict, tuple]:
        return state_dict, tuple()

    def compute_derivatives(
        self,
        state_dict: HubEffectorStateDict,
        integrate_time_step: float,
        rDDot_BN_N: torch.Tensor | None,
        omegaDot_BN_B: torch.Tensor | None,
        sigma_BN: torch.Tensor | None,
        g_N: torch.Tensor,
        back_substitution_matrices: BackSubMatrices,
    ) -> HubEffectorDynamicParams:
        """
        Computes time derivatives of the hub's state, including position, angular velocity, and attitude derivatives.
        Because in back_substitution_matrices, all tensor if zeros but matrix_d and vec_rot

        Args:
            state_dict (HubEffectorStateDict): Dictionary containing the current hub state.
            rDDot_BN_N (torch.Tensor): 3D vector of the hub's second position derivative in the inertial frame (N).
            omegaDot_BN_B (torch.Tensor): 3D vector of the hub's angular acceleration in the body frame (B).
            sigma_BN (torch.Tensor): 3D vector of Modified Rodrigues Parameters (MRP) for hub attitude relative to the inertial frame (N).
            g_N (torch.Tensor): 3D vector of the gravitational acceleration in the inertial frame (N).

        Returns:
            HubEffectorStateDict: Updated dictionary with computed state derivatives.
        """

        dynamic_params = state_dict['dynamic_params']
        velocity = dynamic_params['velocity']
        sigma = dynamic_params['sigma']
        omega = dynamic_params['omega']

        vec_rot = back_substitution_matrices['vec_rot']
        vec_trans = back_substitution_matrices['vec_trans']
        matrix_a = back_substitution_matrices['matrix_a']
        matrix_b = back_substitution_matrices['matrix_b']
        matrix_c = back_substitution_matrices['matrix_c']
        matrix_d = back_substitution_matrices['matrix_d']

        #
        sigma_dot = 0.25 * torch.matmul(
            Bmat(sigma),
            omega.unsqueeze(-1),
        ).squeeze(-1)

        dcm_NB = to_rotation_matrix(sigma)

        intermediate_vector = vec_rot - torch.matmul(
            matrix_c,
            torch.linalg.solve(
                matrix_a,
                vec_trans.unsqueeze(-1),
            ),
        )
        intermediate_matrix = matrix_d - torch.matmul(
            matrix_c,
            torch.linalg.solve(
                matrix_a,
                matrix_b,
            ),
        )

        omega_dot = torch.linalg.solve(
            intermediate_matrix,
            intermediate_vector,
        )
        velocity_dot = torch.matmul(
            dcm_NB,
            torch.linalg.solve(
                matrix_a,
                vec_trans.unsqueeze(-1) - torch.matmul(matrix_b, omega_dot),
            ),
        ).squeeze(-1)
        grav_velocity_dot = g_N
        grav_velocity_bc_dot = g_N
        pos_dot = velocity.clone()

        return HubEffectorDynamicParams(
            pos=pos_dot,
            velocity=velocity_dot,
            sigma=sigma_dot,
            omega=omega_dot,
            grav_velocity=grav_velocity_dot,
            grav_velocity_bc=grav_velocity_bc_dot,
        )

    def update_energy_momentum_contributions(
        self,
        state_dict: HubEffectorStateDict,
        integrate_time_step: float,
        rotAngMomPntCContr_B: torch.Tensor,
        rotEnergyContr: torch.Tensor,
        omega_BN_B: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        moment_of_inertia_matrix_wrt_body_point = state_dict[
            'moment_of_inertia_matrix_wrt_body_point']
        dynamic_params = state_dict['dynamic_params']
        omega = dynamic_params['omega']

        rotAngMomPntCContr_B = moment_of_inertia_matrix_wrt_body_point * omega

        rotEnergyContr = 0.5 * (omega.dot(
            torch.matmul(
                moment_of_inertia_matrix_wrt_body_point,
                omega.unsqueeze(-1),
            ).squeeze(-1)))

        return rotAngMomPntCContr_B, rotEnergyContr

    def modify_states(
        self,
        state_dict: HubEffectorStateDict,
        integrate_time_step: float,
    ) -> HubEffectorStateDict:
        sigma = state_dict['dynamic_params']['sigma']
        sigma_norm = sigma.norm()
        if sigma_norm > 1:
            sigma = -sigma / sigma_norm
            state_dict['dynamic_params']['sigma'] = sigma
            # mrp_switch_count += 1

        return state_dict

    def match_gravity_to_velocity_state(
        self,
        state_dict: HubEffectorStateDict,
        v_CN_N: torch.Tensor,
    ) -> HubEffectorStateDict:
        dynamic_params = state_dict['dynamic_params']

        dynamic_params['grav_velocity'] = dynamic_params['velocity'].clone()
        dynamic_params['grav_velocity_bc'] = v_CN_N
