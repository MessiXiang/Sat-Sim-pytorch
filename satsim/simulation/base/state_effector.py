__all__ = [
    'BackSubMatrices',
    'MassProps',
    'BaseStateEffector',
    'StateEffectorStateDict',
]
from abc import ABC, abstractmethod
from typing import Generic, Mapping, TypedDict, TypeVar, cast

from sympy import integrate
import torch

from satsim.architecture import Module


class BackSubMatrices(TypedDict):
    matrix_a: torch.Tensor
    matrix_b: torch.Tensor
    matrix_c: torch.Tensor
    matrix_d: torch.Tensor
    vec_trans: torch.Tensor
    vec_rot: torch.Tensor


class MassProps(TypedDict):
    mass: torch.Tensor
    moment_of_inertia_matrix_wrt_body_point: torch.Tensor


U = TypeVar('U', bound=Mapping[str, torch.Tensor])


class StateEffectorStateDict(TypedDict, Generic[U]):
    mass_props: MassProps
    dynamic_params: U


T = TypeVar('T', bound=StateEffectorStateDict)


class BaseStateEffector(Module[T], ABC):

    def reset(self) -> T:
        state_dict = super().reset()
        mass_props = dict(
            mass=torch.zeros(1),
            moment_of_inertia_matrix_wrt_body_point=torch.zeros(3, 3),
        )
        state_dict.update(mass_props=mass_props)
        return cast(T, state_dict)

    def update_effector_mass(
        self,
        state_dict: T,
        integrate_time_step: float,
    ) -> MassProps:
        return state_dict['mass_props']

    def update_back_substitution_contribution(
        self,
        state_dict: T,
        integrate_time_step: float,
        back_substitution_contribution: BackSubMatrices,
        sigma_BN: torch.Tensor,
        angular_velocity_BN_B: torch.Tensor,
        g_N: torch.Tensor,
    ) -> BackSubMatrices:
        return back_substitution_contribution

    def update_energy_momentum_contributions(
        self,
        state_dict: T,
        integrate_time_step: float,
        rotAngMomPntCContr_B: torch.Tensor,
        rotEnergyContr: torch.Tensor,
        omega_BN_B: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return rotAngMomPntCContr_B, rotEnergyContr

    def modify_states(
        self,
        state_dict: T,
        integrate_time_step: float,
    ) -> T:
        return state_dict

    def calculate_force_torque_on_body(
        self,
        state_dict: T,
        integrate_time_step: float,
        omega_BN_B: torch.Tensor,
    ) -> T:
        return state_dict

    @abstractmethod
    def compute_derivatives(
        self,
        state_dict: T,
        integrate_time_step: float,
        rDDot_BN_N: torch.Tensor,
        omegaDot_BN_B: torch.Tensor,
        sigma_BN: torch.Tensor,
    ) -> U:
        pass
