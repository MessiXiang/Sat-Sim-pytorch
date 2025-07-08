from re import S
from typing import NotRequired, TypedDict

import torch

from ..base.state_effector import (
    BackSubMatrices,
    StateEffectorMixin,
    StateEffectorStateDict,
)
from satsim.utils import (
    create_skew_symmetric_matrix,
    Bmat,
    to_rotation_matrix,
)


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
    mass: float
    hub_inertia_matrix: torch.Tensor
    position_center_of_mass: torch.Tensor
    position_primary_hub_center_of_mass: NotRequired[torch.Tensor]
    IHubPntBc_P: NotRequired[torch.Tensor]
    hub_back_substitution_matrices: BackSubMatrices


# MRPSwitchCount deserted
class HubEffector(StateEffectorMixin[HubEffectorStateDict]):

    def __init__(
        self,
        mass: float = 1.,
        hub_inertia_matrix: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
        omega: torch.Tensor | None = None,
    ) -> None:
        """
        Initialize a spacecraft dynamics object with physical and kinematic parameters.

        Args:
            mass (float, optional): Mass of the spacecraft in kilograms. Defaults to 1.0.
            position_center_of_mass (torch.Tensor, optional): Position vector of the spacecraft's center of mass
                relative to the body frame origin, in body frame coordinates (3x1 tensor).
                Defaults to zero vector if None.
            hub_inertia_matrix (torch.Tensor, optional): Inertia tensor of the spacecraft about the
                center of mass, in body frame coordinates (3x3 tensor). Defaults to identity
                matrix if None.
            pos (torch.Tensor, optional): Initial position vector of the spacecraft's
                center of mass relative to the inertial frame origin, in inertial frame
                coordinates (3x1 tensor). Defaults to zero vector if None.
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
        self.mass = mass
        self.position_center_of_mass = position_center_of_mass or torch.zeros(
            3)
        self.hub_inertia_matrix = hub_inertia_matrix or torch.eye(3, 3)
        self.pos = torch.zeros(3) if pos is None else pos
        self.velocity = velocity or torch.zeros(3)
        self.sigma = sigma or torch.zeros(3)
        self.omega = omega or torch.zeros(3)

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
            hub_inertia_matrix=self.hub_inertia_matrix.clone(),
            position_center_of_mass=self.position_center_of_mass.clone(),
        )

    def update_effector_mass(
        self,
        state_dict: HubEffectorStateDict,
    ) -> HubEffectorStateDict:
        effProps = state_dict['effProps']
        mass = state_dict['mass']
        r_BP_P = state_dict['r_BP_P']
        dcm_BP = state_dict['dcm_BP']
        position_center_of_mass = state_dict['position_center_of_mass']
        hub_inertia_matrix = state_dict['hub_inertia_matrix']

        position_primary_hub_center_of_mass = r_BP_P + dcm_BP.transpose(
        ) @ position_center_of_mass
        IHubPntBc_P = dcm_BP.transpose() @ hub_inertia_matrix @ dcm_BP

        effProps[
            'IEffPntB_B'] = IHubPntBc_P + mass * create_skew_symmetric_matrix(
                position_primary_hub_center_of_mass
            ) * create_skew_symmetric_matrix(
                position_primary_hub_center_of_mass).transpose()

        effProps['rEff_CB_B'] = position_primary_hub_center_of_mass
        effProps['rEffPrime_CB_B'] = torch.zeros_like(
            effProps['rEffPrime_CB_B'])
        effProps['IEffPrimePntB_B'] = torch.zeros_like(
            effProps['IEffPrimePntB_B'])

        state_dict['effProps'] = effProps
        state_dict[
            'position_primary_hub_center_of_mass'] = position_primary_hub_center_of_mass
        state_dict['IHubPntBc_P'] = IHubPntBc_P

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
        position_primary_hub_center_of_mass = state_dict[
            'position_primary_hub_center_of_mass']
        mass = state_dict['mass']
        IHubPntBc_P = state_dict['IHubPntBc_P']
        dynamic_params = state_dict['dynamic_params']
        omega = dynamic_params['omega']

        rDot_BcB_B = omega.cross(position_primary_hub_center_of_mass)
        rotAngMomPntCContr_B = IHubPntBc_P * omega + mass * position_primary_hub_center_of_mass.cross(
            rDot_BcB_B)

        rotEnergyContr = 0.5 * (omega.dot(IHubPntBc_P @ omega) +
                                mass * rDot_BcB_B.dot(rDot_BcB_B))

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
    ):
        dynamic_params = state_dict['dynamic_params']

        dynamic_params['grav_velocity'] = dynamic_params['velocity'].clone()
        dynamic_params['grav_velocity_bc'] = v_CN_N
