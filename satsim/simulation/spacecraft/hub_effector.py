__all__ = [
    'HubEffector',
    'HubEffectorDynamicParams',
    'HubEffectorStateDict',
]
from typing import TypedDict

import torch

from satsim.utils import Bmat, mrp_to_rotation_matrix

from ..base.state_effector import (BackSubMatrices, BaseStateEffector,
                                   MassProps, StateEffectorStateDict)


class HubEffectorDynamicParams(TypedDict):
    position_BP_N: torch.Tensor  # [3]
    velocity_BP_N: torch.Tensor  # [3]
    attitude_BN: torch.Tensor  # [3]
    angular_velocity_BN_B: torch.Tensor  # [3]


HubEffectorStateDict = StateEffectorStateDict[HubEffectorDynamicParams]


# MRPSwitchCount deserted
class HubEffector(
        BaseStateEffector[HubEffectorStateDict], ):

    def __init__(
        self,
        *args,
        mass: torch.Tensor,
        moment_of_inertia_matrix_wrt_body_point: torch.Tensor,
        position_BP_N: torch.Tensor,
        velocity_BP_N: torch.Tensor,
        attitude_BN: torch.Tensor | None = None,
        angular_velocity_BN_B: torch.Tensor | None = None,
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

        self._position_BP_N_init = position_BP_N
        self._velocity_BP_N_init = velocity_BP_N
        self._attitude_BN_init = torch.zeros(
            3) if attitude_BN is None else attitude_BN
        self._angular_velocity_BN_B_init = torch.zeros(
            3) if angular_velocity_BN_B is None else angular_velocity_BN_B

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
            position_BP_N=self._position_BP_N_init.clone(),
            velocity_BP_N=self._velocity_BP_N_init.clone(),
            attitude_BN=self._attitude_BN_init.clone(),
            angular_velocity_BN_B=self._angular_velocity_BN_B_init.clone(),
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
        '''
        Computes the time derivatives of the hub's state.

        Args:
            state_dict (HubEffectorStateDict): Dictionary containing the current hub state.
            integrate_time_step (float): Integration time step.
            back_substitution_matrices (BackSubMatrices): Matrices for back substitution.
            gravity_acceleration (torch.Tensor): Gravitational acceleration in inertial frame.
            spacecraft_mass (torch.Tensor): Mass of the spacecraft.

        Returns:
            HubEffectorDynamicParams: Updated dynamic parameters with computed state derivatives.
        '''
        dynamic_params = state_dict['dynamic_params']
        velocity_BP_N = dynamic_params['velocity_BP_N']
        attitude_BN = dynamic_params['attitude_BN']
        angular_velocity_BN_B = dynamic_params['angular_velocity_BN_B']

        ext_torque_B_B = back_substitution_matrices['ext_torque_B_B']
        ext_force_B_B = back_substitution_matrices['ext_force_B_B']
        moment_of_inertia_matrix = back_substitution_matrices[
            'moment_of_inertia_matrix']

        attitude_dot = 0.25 * torch.einsum(
            '...ij,...j->...i',
            Bmat(attitude_BN),
            angular_velocity_BN_B,
        )

        direction_cosine_matrix_BN = mrp_to_rotation_matrix(attitude_BN)

        angular_velocity_dot: torch.Tensor = torch.linalg.solve(
            moment_of_inertia_matrix,
            ext_torque_B_B.unsqueeze(-1),
        )
        velocity_dot = torch.einsum(
            '...ij,...i->...j',
            direction_cosine_matrix_BN,
            (ext_force_B_B / spacecraft_mass.unsqueeze(-1)),
        ) + gravity_acceleration
        position_dot = velocity_BP_N.clone()

        return HubEffectorDynamicParams(
            position_BP_N=position_dot,
            velocity_BP_N=velocity_dot.squeeze(-1),
            attitude_BN=attitude_dot,
            angular_velocity_BN_B=angular_velocity_dot.squeeze(-1),
        )

    def normalize_attitude(
        self,
        state_dict: HubEffectorStateDict,
    ) -> HubEffectorStateDict:
        attitude_BN = state_dict['dynamic_params']['attitude_BN']
        attitude_BN_norm = attitude_BN.norm(dim=-1, keepdim=True)
        normalize_mask = attitude_BN_norm > 1
        if torch.any(normalize_mask):
            attitude_BN = torch.where(
                normalize_mask,
                -attitude_BN /
                (attitude_BN * attitude_BN).sum(-1, keepdim=True),
                attitude_BN,
            )
        state_dict['dynamic_params']['attitude_BN'] = attitude_BN

        return state_dict
