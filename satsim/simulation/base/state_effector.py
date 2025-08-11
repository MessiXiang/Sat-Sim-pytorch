__all__ = [
    'BackSubMatrices',
    'MassProps',
    'BaseStateEffector',
    'StateEffectorStateDict',
]
from abc import ABC, abstractmethod
from typing import Generic, Mapping, TypedDict, TypeVar, cast

import torch

from satsim.architecture import Module


class BackSubMatrices(TypedDict):
    # Because we have simplified satellite as a mass point rather than a rigid body
    # its equation of motion become more simple
    moment_of_inertia_matrtix: torch.Tensor
    ext_force: torch.Tensor
    ext_torque: torch.Tensor


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
        *args,
        **kwargs,
    ) -> BackSubMatrices:
        return back_substitution_contribution

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
        *args,
        **kwargs,
    ) -> T:
        return state_dict

    @abstractmethod
    def compute_derivatives(
        self,
        state_dict: T,
        integrate_time_step: float,
        *args,
        **kwargs,
    ) -> U:
        pass
