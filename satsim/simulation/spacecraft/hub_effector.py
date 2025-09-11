__all__ = [
    'HubEffector',
    'HubEffectorDynamicParams',
    'HubEffectorStateDict',
]
from typing import TypedDict

import torch
from todd.loggers import master_logger

from satsim.utils import Bmat, mrp_to_rotation_matrix

from ..base.state_effector import BackSubMatrices, MassProps, BaseStateEffector, StateEffectorStateDict


class HubEffectorDynamicParams(TypedDict):
    position: torch.Tensor  # [3]
    velocity: torch.Tensor  # [3]
    attitude: torch.Tensor  # [3]
    angular_velocity: torch.Tensor  # [3]
    grav_velocity: torch.Tensor  # [3]


HubEffectorStateDict = StateEffectorStateDict[HubEffectorDynamicParams]


# MRPSwitchCount deserted
class HubEffector(
        BaseStateEffector[HubEffectorStateDict], ):

    def __init__(
        self,
        *args,
        mass: torch.Tensor,
        moment_of_inertia_matrix_wrt_body_point: torch.Tensor,
        position: torch.Tensor,
        velocity: torch.Tensor,
        attitude: torch.Tensor | None = None,
        angular_velocity: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer(
            '_mass',
            mass,
            persistent=False,
        )
        self.register_buffer(
            '_moment_of_inertia_matrix_wrt_body_point',
            moment_of_inertia_matrix_wrt_body_point,
            persistent=False,
        )

        self._position_init = position
        self._velocity_init = velocity
        self._attitude_init = torch.zeros(3) if attitude is None else attitude
        self._angular_velocity_init = torch.zeros(
            3) if angular_velocity is None else angular_velocity

    @property
    def mass(self) -> torch.Tensor:
        return self.get_buffer('_mass')

    @property
    def moment_of_inertia_matrix_wrt_body_point(self) -> torch.Tensor:
        return self.get_buffer('_moment_of_inertia_matrix_wrt_body_point')

    def reset(self) -> HubEffectorStateDict:
        state_dict = super().reset()

        mass_props: MassProps = dict(
            mass=self.mass,
            moment_of_inertia_matrix_wrt_body_point=self.
            moment_of_inertia_matrix_wrt_body_point,
        )

        dynamic_params = HubEffectorDynamicParams(
            position=self._position_init.clone(),
            velocity=self._velocity_init.clone(),
            attitude=self._attitude_init.clone(),
            angular_velocity=self._angular_velocity_init.clone(),
            grav_velocity=self._velocity_init.clone(),
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
        back_substitution_matrices: BackSubMatrices,
        gravity_acceleration: torch.Tensor,
        spacecraft_mass: torch.Tensor,
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
        attitude = dynamic_params['attitude']
        angular_velocity = dynamic_params['angular_velocity']

        ext_torque = back_substitution_matrices['ext_torque']
        ext_force = back_substitution_matrices['ext_force']
        moment_of_inertia_matrix = back_substitution_matrices[
            'moment_of_inertia_matrix']

        attitude_dot = 0.25 * torch.matmul(
            Bmat(attitude),
            angular_velocity.unsqueeze(-1),
        ).squeeze(-1)

        dcm_NB = mrp_to_rotation_matrix(attitude)

        angular_velocity_dot: torch.Tensor = torch.linalg.solve(
            moment_of_inertia_matrix,
            ext_torque.unsqueeze(-1),
        )
        velocity_dot = torch.matmul(
            dcm_NB,
            (ext_force / spacecraft_mass.unsqueeze(-1)).unsqueeze(-1),
        ).squeeze(-1) + gravity_acceleration
        grav_velocity_dot = gravity_acceleration
        position_dot = velocity.clone()

        return HubEffectorDynamicParams(
            position=position_dot,
            velocity=velocity_dot.squeeze(-1),
            attitude=attitude_dot,
            angular_velocity=angular_velocity_dot.squeeze(-1),
            grav_velocity=grav_velocity_dot,
        )

    def normalize_attitude(
        self,
        state_dict: HubEffectorStateDict,
    ) -> HubEffectorStateDict:
        sigma = state_dict['dynamic_params']['attitude']
        sigma_norm = sigma.norm(dim=-1, keepdim=True)
        normalize_mask = sigma_norm > 1

        if torch.any(normalize_mask):
            master_logger.warning(
                "The norm of MRP is greater than 1. Normalizing it.")

            sigma = torch.where(
                normalize_mask,
                -sigma /
                torch.einsum('...i,...i->...', sigma, sigma).unsqueeze(-1),
                sigma,
            )
            state_dict['dynamic_params']['attitude'] = sigma

        return state_dict

    def match_gravity_to_velocity_state(
        self,
        state_dict: HubEffectorStateDict,
    ) -> HubEffectorStateDict:
        dynamic_params = state_dict['dynamic_params']

        dynamic_params['grav_velocity'] = dynamic_params['velocity'].clone()
        return state_dict
