from re import S
from typing import NotRequired, TypedDict

import torch

from ..base.state_effector import (
    BackSubMatrices,
    StateEffectorMixin,
    StateEffectorStateDict,
)
from satsim.utils import Bmat, to_rotation_matrix


class HubDynamicParams(TypedDict):
    pos: torch.Tensor  # [3]
    velocity: torch.Tensor  # [3]
    sigma: torch.Tensor  # [3]
    omega: torch.Tensor  # [3]
    grav_velocity: torch.Tensor  # [3]
    grav_velocity_bc: torch.Tensor  # [3]
    pos_dot: NotRequired[torch.Tensor]  # [3]
    velocity_dot: NotRequired[torch.Tensor]  # [3]
    sigma_dot: NotRequired[torch.Tensor]  # [3]
    omega_dot: NotRequired[torch.Tensor]  # [3]
    grav_velocity_dot: NotRequired[torch.Tensor]  # [3]
    grav_velocity_bc_dot: NotRequired[torch.Tensor]  # [3]


class HubEffectorStateDict(StateEffectorStateDict):
    dynamic_params: HubDynamicParams
    mass: torch.Tensor
    hub_moment_of_inertia_matrix_wrt_body_point: torch.Tensor


# MRPSwitchCount deserted
class HubEffector(StateEffectorMixin[HubEffectorStateDict]):

    def __init__(
        self,
        mass: torch.Tensor | None = None,
        hub_moment_of_inertia_matrix_wrt_body_point: torch.Tensor
        | None = None,
        pos: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
        omega: torch.Tensor | None = None,
    ) -> None:
        """
        Initialize a spacecraft dynamics object with physical and kinematic parameters.

        Args:
            mass (float, optional): Mass of the spacecraft in kilograms. Defaults to 1.0.
            hub_moment_of_inertia_matrix_wrt_body_point (torch.Tensor, optional): Inertia tensor of the spacecraft about the center of mass, in body frame coordinates (3x3 tensor). Defaults to identity matrix if None.
            pos (torch.Tensor, optional): Initial position vector of the spacecraft's center of mass relative to the inertial frame origin, in inertial frame coordinates (3x1 tensor). Defaults to zero vector if None.
            velocity (torch.Tensor, optional): Initial velocity vector of the spacecraft's
                center of mass in the inertial frame (3x1 tensor). Defaults to zero vector if None.
            sigma (torch.Tensor, optional): Initial Modified Rodrigues Parameters (MRP)
                representing the attitude of the body frame relative to the inertial frame
                (3x1 tensor). Defaults to zero vector if None.
            omega (torch.Tensor, optional): Initial angular velocity of the body frame
                relative to the inertial frame, in body frame coordinates (3x1 tensor).
                Defaults to zero vector if None.

        Returns:
            None
        """
        self.mass = torch.tensor([1.]) if mass is None else mass
        self.hub_moment_of_inertia_matrix_wrt_body_point = torch.eye(
            3, 3
        ) if hub_moment_of_inertia_matrix_wrt_body_point is None else hub_moment_of_inertia_matrix_wrt_body_point
        self.pos = torch.zeros(3) if pos is None else pos
        self.velocity = torch.zeros(3) if velocity is None else velocity
        self.sigma = torch.zeros(3) if sigma is None else sigma
        self.omega = torch.zeros(3) if omega is None else omega

    def reset(self) -> HubEffectorStateDict:
        state_dict = super().state_effector_reset()
        state_dict['effProps']['mEff'] = self.mass
        dynamic_params = HubDynamicParams(
            pos=self.pos.clone(),
            velocity=self.velocity.clone(),
            sigma=self.sigma.clone(),
            omega=self.omega.clone(),
            grav_velocity=self.velocity.clone(),
            grav_velocity_bc=self.velocity.clone(),
        )
        return dict(
            **state_dict,
            dynamic_params=dynamic_params,
            mass=self.mass,
            hub_moment_of_inertia_matrix_wrt_body_point=self.
            hub_moment_of_inertia_matrix_wrt_body_point.clone(),
        )

    def update_effector_mass(
        self,
        state_dict: HubEffectorStateDict,
    ) -> HubEffectorStateDict:
        state_dict['effProps']['IEffPntB_B'] = state_dict[
            'hub_moment_of_inertia_matrix_wrt_body_point'].clone()

        return state_dict

    def compute_derivatives(
        self,
        state_dict: HubEffectorStateDict,
        rDDot_BN_N: torch.Tensor | None,
        omegaDot_BN_B: torch.Tensor | None,
        sigma_BN: torch.Tensor | None,
        g_N: torch.Tensor,
        back_substitution_matrices: BackSubMatrices,
    ) -> HubEffectorStateDict:
        """
        Computes time derivatives of the hub's state, including position, angular velocity, and attitude derivatives.

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
        matrix_a_inverse = matrix_a.inverse()
        matrix_b = back_substitution_matrices['matrix_b']
        matrix_c = back_substitution_matrices['matrix_c']
        matrix_d = back_substitution_matrices['matrix_d']

        #
        dynamic_params['sigma_dot'] = 0.25 * Bmat(sigma) @ omega

        dcm_NB = to_rotation_matrix(sigma)

        intermediate_vector = vec_rot - matrix_c @ matrix_a_inverse @ vec_trans
        intermediate_matrix = matrix_d - matrix_c @ matrix_a_inverse @ matrix_b

        omega_dot = intermediate_matrix.inverse() @ intermediate_vector
        dynamic_params['omega_dot'] = omega_dot
        dynamic_params['velocity_dot'] = dcm_NB @ matrix_a_inverse @ (
            vec_trans - matrix_b @ omega_dot)
        dynamic_params['grav_velocity_dot'] = g_N
        dynamic_params['grav_velocity_bc_dot'] = g_N
        dynamic_params['pos_dot'] = velocity.clone()

        state_dict['dynamic_params'] = dynamic_params

        return state_dict

    def update_energy_momentum_contributions(
        self,
        state_dict: HubEffectorStateDict,
        rotAngMomPntCContr_B: torch.Tensor,
        rotEnergyContr: torch.Tensor,
        omega_BN_B: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hub_moment_of_inertia_matrix_wrt_body_point = state_dict[
            'hub_moment_of_inertia_matrix_wrt_body_point']
        dynamic_params = state_dict['dynamic_params']
        omega = dynamic_params['omega']

        rotAngMomPntCContr_B = hub_moment_of_inertia_matrix_wrt_body_point * omega

        rotEnergyContr = 0.5 * (omega.dot(
            hub_moment_of_inertia_matrix_wrt_body_point @ omega))

        return rotAngMomPntCContr_B, rotEnergyContr

    def modify_states(
            self, state_dict: HubEffectorStateDict) -> HubEffectorStateDict:
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
