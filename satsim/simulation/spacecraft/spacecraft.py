__all__ = [
    'Spacecraft',
    'SpacecraftStateDict',
    'SpacecraftStateOutput',
    'DynamicParamsDict',
]
from copy import deepcopy
from typing import TypedDict

import torch

from satsim.architecture import Module

from ...utils.matrix_support import to_rotation_matrix
from ..base.state_effector import (BackSubMatrices, BaseStateEffector,
                                   MassProps, StateEffectorStateDict)
from ..gravity.gravity_effector import Ephemeris, GravityField
from .hub_effector import HubEffector, HubEffectorStateDict


class SpacecraftStateOutput(TypedDict):
    position_in_inerital: torch.Tensor
    velocity_in_inertial: torch.Tensor
    sigma: torch.Tensor
    omega: torch.Tensor
    omega_dot: torch.Tensor
    total_accumulated_non_gravitational_velocity_change_in_body: torch.Tensor
    total_accumulated_non_gravitational_velocity_change_in_inertial: torch.Tensor
    non_conservative_acceleration_of_body_in_body: torch.Tensor


class SpacecraftStateDict(TypedDict):
    mass_props: MassProps
    _hub: HubEffectorStateDict
    accumulated_non_gravitational_velocity_change_in_body: torch.Tensor
    # state effectors are stored as submodules
    # _state_effector_i: StateEffectorStateDict


DynamicParamsDict = dict[str, dict[str, torch.Tensor]]


# MRPSwitchCount deserted
class Spacecraft(
        Module[SpacecraftStateDict], ):

    def __init__(
        self,
        *args,
        mass: torch.Tensor | None = None,
        moment_of_inertia_matrix_wrt_body_point: torch.Tensor
        | None = None,
        pos: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
        omega: torch.Tensor | None = None,
        gravity_field: GravityField,
        state_effectors: list[BaseStateEffector] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._hub = HubEffector(
            mass=mass,
            moment_of_inertia_matrix_wrt_body_point=
            moment_of_inertia_matrix_wrt_body_point,
            pos=pos,
            velocity=velocity,
            sigma=sigma,
            omega=omega,
        )
        self.add_module('_gravity_field', gravity_field)
        state_effectors = state_effectors or []
        self._num_state_effectors = len(state_effectors)
        for id, state_effector in enumerate(state_effectors or []):
            self.add_module(f'_state_effector_{id}', state_effector)

    @property
    def _gravity_field(self) -> GravityField:
        return self.get_submodule('_gravity_field')

    def reset(self) -> SpacecraftStateDict:
        state_dict = super().reset()

        num_planet_bodies = self._gravity_field.num_planet_bodies + 1
        fake_central_body_ephemeris = Ephemeris(
            position_in_inertial=torch.zeros(1, 3),
            velocity_in_inertial=torch.zeros(1, 3),
        )
        fake_planet_bodies_ephemeris = Ephemeris(
            position_in_inertial=torch.zeros(num_planet_bodies, 3),
            velocity_in_inertial=torch.zeros(num_planet_bodies, 3),
        )
        state_dict = self.equation_of_motion(
            state_dict=state_dict,
            central_body_ephemeris=fake_central_body_ephemeris,
            planet_body_ephemeris=fake_planet_bodies_ephemeris,
        )

        return state_dict

    def update_spacecraft_mass_props(
        self,
        state_dict: SpacecraftStateDict,
        integrate_time_step: float,
    ) -> SpacecraftStateDict:
        """ The mass of spacecraft can be potentially changed due to fuel consumption or other reasons. 
        But now there is no such feature implemented. This method is left for future use.
        This method updates the mass properties of the spacecraft by summing up the mass properties of the
        hub and all state effectors.
        """
        mass = []
        moment_of_inertia_matrix_wrt_body_point = []
        for id in range(self._num_state_effectors):
            state_effector_id = f'_state_effector_{id}'
            state_effector: BaseStateEffector = self.get_submodule(
                state_effector_id)
            state_effector_state_dict: StateEffectorStateDict = state_dict.get(
                state_effector_id, {})
            state_effector_mass_props = state_effector.update_effector_mass(
                state_dict=state_effector_state_dict,
                integrate_time_step=integrate_time_step,
            )
            mass.append(state_effector_mass_props['mass'])
            moment_of_inertia_matrix_wrt_body_point.append(
                state_effector_mass_props[
                    'moment_of_inertia_matrix_wrt_body_point'])

        mass = torch.stack(mass, dim=-1).sum(dim=-1)
        moment_of_inertia_matrix_wrt_body_point = torch.stack(
            moment_of_inertia_matrix_wrt_body_point, dim=-1).sum(dim=-1)
        mass = mass + self._hub.get_buffer('mass')
        moment_of_inertia_matrix_wrt_body_point = (
            moment_of_inertia_matrix_wrt_body_point +
            self._hub.get_buffer('moment_of_inertia_matrix_wrt_body_point'))

        state_dict['mass_props'] = dict(
            mass=mass,
            moment_of_inertia_matrix_wrt_body_point=
            moment_of_inertia_matrix_wrt_body_point,
        )
        return state_dict

    def equation_of_motion(
        self,
        state_dict: SpacecraftStateDict,
        integrate_time_step: float,
        central_body_ephemeris: Ephemeris,
        planet_body_ephemeris: Ephemeris,
    ) -> tuple[DynamicParamsDict, DynamicParamsDict]:
        state_dict = self.update_spacecraft_mass_props(
            state_dict,
            integrate_time_step=integrate_time_step,
        )
        mass = state_dict['mass_props']['mass']
        moment_of_inertia_matrix_wrt_body_point = state_dict['mass_props'][
            'moment_of_inertia_matrix_wrt_body_point']

        hub_state_dict: HubEffectorStateDict = state_dict.get('_hub', {})
        position = hub_state_dict['dynamic_params']['pos']
        sigma = hub_state_dict['dynamic_params']['sigma']
        omega = hub_state_dict['dynamic_params']['omega']

        direction_cosine_matrix_body_to_inertial = to_rotation_matrix(sigma)

        gravity_acceleration = self._gravity_field.compute_gravity_field(
            position,
            central_body_ephemeris,
            planet_body_ephemeris,
        )

        # calculate back substitution matrices
        back_substitution_contribution = BackSubMatrices(
            matrix_d=torch.zeros_like(moment_of_inertia_matrix_wrt_body_point),
            matrix_a=torch.zeros_like(moment_of_inertia_matrix_wrt_body_point),
            matrix_b=torch.zeros_like(moment_of_inertia_matrix_wrt_body_point),
            matrix_c=torch.zeros_like(moment_of_inertia_matrix_wrt_body_point),
            vec_trans=torch.zeros_like(position),
            vec_rot=torch.zeros_like(position),
        )
        # update back substitution matrices for state effectors
        for id in range(self._num_state_effectors):
            state_effector_id = f'_state_effector_{id}'
            state_effector: BaseStateEffector = self.get_submodule(
                state_effector_id)
            state_effector_state_dict: StateEffectorStateDict = state_dict.get(
                state_effector_id,
                dict(),
            )
            back_substitution_contribution = state_effector.update_back_substitution_contribution(
                state_dict=state_effector_state_dict,
                integrate_time_step=integrate_time_step,
                back_substitution_contribution=back_substitution_contribution,
                sigma_BN=sigma,
                angular_velocity_BN_B=omega,
                g_N=gravity_acceleration,
            )

        # update back substitution matrices for hub
        back_substitution_contribution['matrix_a'] += back_substitution_contribution['matrix_a'] + \
            mass.unsqueeze(-1) * torch.eye(3)
        back_substitution_contribution['matrix_d'] = back_substitution_contribution['matrix_d'] + \
            moment_of_inertia_matrix_wrt_body_point
        back_substitution_contribution['vec_rot'] = back_substitution_contribution['vec_rot'] + \
            torch.cross(
                torch.matmul(
                    moment_of_inertia_matrix_wrt_body_point,
                    omega.unsqueeze(-1),
                ).squeeze(-1),
                omega,
                dim=-1,
            )

        # update back substitution matrices for gravity field
        gravity_force_in_inertial = gravity_acceleration * mass
        gravity_force_in_body = torch.matmul(
            direction_cosine_matrix_body_to_inertial.transpose(-1, -2),
            gravity_force_in_inertial.unsqueeze(-1),
        ).squeeze(-1)
        back_substitution_contribution['vec_trans'] = back_substitution_contribution['vec_trans'] + \
            gravity_force_in_body

        hub_state_dot = self._hub.compute_derivatives(
            state_dict=hub_state_dict,
            integrate_time_step=integrate_time_step,
            rDDot_BN_N=hub_state_dict['dynamic_params']['velocity_dot'],
            omegaDot_BN_B=hub_state_dict['dynamic_params']['omega_dot'],
            sigma_BN=sigma,
            g_N=gravity_acceleration,
            back_substitution_matrices=back_substitution_contribution,
        )

        states_dot = dict(_hub=hub_state_dot)
        for id in range(self._num_state_effectors):
            state_effector_id = f'_state_effector_{id}'
            state_effector: BaseStateEffector = self.get_submodule(
                state_effector_id)
            state_effector_state_dict: StateEffectorStateDict = state_dict.get(
                state_effector_id,
                dict(),
            )
            state_dot = state_effector.compute_derivatives(
                state_dict=state_effector_state_dict,
                integrate_time_step=integrate_time_step,
                rDDot_BN_N=hub_state_dict['dynamic_params']['pos_dot'],
                omegaDot_BN_B=hub_state_dict['dynamic_params']['omega_dot'],
                sigma_BN=sigma,
            )
            states_dot[state_effector_id] = state_dot

        return states_dot

    def get_dynamic_params(
        self,
        state_dict: SpacecraftStateDict,
    ) -> DynamicParamsDict:
        dynamic_params = dict()
        hub_state_dict: HubEffectorStateDict = state_dict.get('_hub', {})
        dynamic_params['_hub'] = hub_state_dict['dynamic_params']

        for id in range(self._num_state_effectors):
            state_effector_id = f'_state_effector_{id}'
            state_effector_state_dict = state_dict.get(state_effector_id, {})
            dynamic_params[state_effector_id] = (
                state_effector_state_dict['dynamic_params'])

        return dynamic_params

    def apply_dynamic_params(
        self,
        state_dict: SpacecraftStateDict,
        dynamic_params: DynamicParamsDict,
    ) -> SpacecraftStateDict:
        hub_state = dynamic_params['_hub']
        state_dict['_hub']['dynamic_params'] = hub_state

        for id in range(self._num_state_effectors):
            state_effector_id = f'_state_effector_{id}'
            state = dynamic_params[state_effector_id]
            state_dict[state_effector_id]['dynamic_params'] = state

        return state_dict

    def integrate_to_this_time(
        self,
        state_dict: SpacecraftStateDict,
        central_body_ephemeris: Ephemeris,
        planet_body_ephemeris: Ephemeris | None = None,
    ) -> SpacecraftStateDict:
        """ Using 4 stage rungekutta method to solve the next state of the 
        spacecraft. Originally, this method need to determine some matrix of coeificient, but here we just use a specified set of values for simplification. 
        Additionally, in future, we can define a new differential
        equation solver class to make it possible for user to write their own
        differential equation solver. 
        """
        # get dynamic params space and save current dynamic params state
        dynamic_params = self.get_dynamic_params(state_dict)
        previous_dynamic_params = deepcopy(dynamic_params)

        # stage 1
        k1 = self.equation_of_motion(
            state_dict=state_dict,
            integrate_time_step=-1. * self._timer.dt,
            central_body_ephemeris=central_body_ephemeris,
            planet_body_ephemeris=planet_body_ephemeris,
        )
        #save current dynamic state

        # stage 2
        for module_name, module_dynamic_params in dynamic_params.items():
            module_dynamic_params_dot = k1[module_name]
            module_previous_dynamic_state = previous_dynamic_params[
                module_name]
            for state_name in module_dynamic_params.keys():
                module_dynamic_params[state_name] = module_previous_dynamic_state[state_name] + \
                        0.5 * self._timer.dt * \
                        module_dynamic_params_dot[state_name]
        state_dict = self.apply_dynamic_params(state_dict, dynamic_params)
        k2 = self.equation_of_motion(
            state_dict=state_dict,
            integrate_time_step=-0.5 * self._timer.dt,
            central_body_ephemeris=central_body_ephemeris,
            planet_body_ephemeris=planet_body_ephemeris,
        )

        # state 3
        for module_name, module_dynamic_params in dynamic_params.items():
            module_dynamic_params_dot = k2[module_name]
            module_previous_dynamic_state = previous_dynamic_params[
                module_name]
            for state_name in module_dynamic_params.keys():
                module_dynamic_params[state_name] = module_previous_dynamic_state[state_name] + \
                        0.5 * self._timer.dt * \
                        module_dynamic_params_dot[state_name]
        state_dict = self.apply_dynamic_params(state_dict, dynamic_params)
        k3 = self.equation_of_motion(
            state_dict=state_dict,
            integrate_time_step=-0.5 * self._timer.dt,
            central_body_ephemeris=central_body_ephemeris,
            planet_body_ephemeris=planet_body_ephemeris,
        )

        # stage 4
        for module_name, module_dynamic_params in dynamic_params.items():
            module_dynamic_params_dot = k3[module_name]
            module_previous_dynamic_state = previous_dynamic_params[
                module_name]
            for state_name in module_dynamic_params.keys():
                module_dynamic_params[state_name] = module_previous_dynamic_state[state_name] + \
                        1 * self._timer.dt * \
                        module_dynamic_params_dot[state_name]
        state_dict = self.apply_dynamic_params(state_dict, dynamic_params)
        k4 = self.equation_of_motion(
            state_dict=state_dict,
            integrate_time_step=0. * self._timer.dt,
            central_body_ephemeris=central_body_ephemeris,
            planet_body_ephemeris=planet_body_ephemeris,
        )

        # now we can calculate the next state
        for module_name, module_dynamic_params in dynamic_params.items():
            module_state_dot_k1 = k1[module_name]
            module_state_dot_k2 = k2[module_name]
            module_state_dot_k3 = k3[module_name]
            module_state_dot_k4 = k4[module_name]
            module_previous_dynamic_state = previous_dynamic_params[
                module_name]
            for state_name in module_dynamic_params.keys():
                module_dynamic_params[state_name] = module_previous_dynamic_state[state_name] + \
                    1/6 * self._timer.dt * module_state_dot_k1[state_name] + \
                    1/3 * self._timer.dt * module_state_dot_k2[state_name] + \
                    1/3 * self._timer.dt * module_state_dot_k3[state_name] + \
                    1/6 * self._timer.dt * module_state_dot_k4[state_name]
        state_dict = self.apply_dynamic_params(state_dict, dynamic_params)
        return state_dict

    def update_state_effectors(
        self,
        state_dict: SpacecraftStateDict,
        *args,
        **kwargs,
    ) -> SpacecraftStateDict:
        for id in range(self._num_state_effectors):
            state_effector_id = f'_state_effector_{id}'
            state_effector: BaseStateEffector = self.get_submodule(
                state_effector_id)
            state_effector_state_dict: StateEffectorStateDict = state_dict.get(
                state_effector_id, {})
            state_effector_state_dict, _ = state_effector(
                state_effector_state_dict, *args, **kwargs)
            state_dict[state_effector_id] = state_effector_state_dict

        return state_dict

    def forward(
        self,
        state_dict: SpacecraftStateDict,
        *args,
        central_body_ephemeris: Ephemeris,
        planet_body_ephemeris: Ephemeris | None = None,
        **kwargs,
    ) -> tuple[SpacecraftStateDict, tuple[SpacecraftStateOutput]]:
        ## pre-solution
        hub_state_dict: HubEffectorStateDict = state_dict.get('_hub', {})
        hub_state_dict = self._hub.match_gravity_to_velocity_state(
            hub_state_dict,
            hub_state_dict['dynamic_params']['velocity'],
        )
        state_dict['_hub'] = hub_state_dict
        omega_before = hub_state_dict['dynamic_params']['omega'].clone()

        ## solve next state
        state_dict = self.integrate_to_this_time(
            state_dict=state_dict,
            central_body_ephemeris=central_body_ephemeris,
            planet_body_ephemeris=planet_body_ephemeris,
        )

        ## post-solution
        state_dict = self.update_spacecraft_mass_props(
            state_dict,
            integrate_time_step=0. * self._timer.dt,
        )
        velocity = state_dict['_hub']['dynamic_params']['velocity']
        sigma = state_dict['_hub']['dynamic_params']['sigma']
        direction_cosine_matrix_body_to_inertial = to_rotation_matrix(sigma)
        gravitational_velocity = state_dict['_hub']['dynamic_params'][
            'grav_velocity']
        state_dict[
            'accumulated_non_gravitational_velocity_change_in_body'] = state_dict[
                'accumulated_non_gravitational_velocity_change_in_body'] + torch.matmul(
                    direction_cosine_matrix_body_to_inertial.transpose(-1, -2),
                    (velocity - gravitational_velocity).unsqueeze(-1),
                ).squeeze(-1)
        state_dict['accumulated_non_gravitational_velocity_change_in_inertial'] = state_dict[
            'accumulated_non_gravitational_velocity_change_in_inertial'] + \
                (velocity - gravitational_velocity)

        non_conservative_acceleration_of_body_in_body = torch.matmul(
            direction_cosine_matrix_body_to_inertial.transpose(-1, -2),
            (velocity - gravitational_velocity).unsqueeze(-1),
        ).squeeze(-1) / self._timer.dt

        omega = state_dict['_hub']['dynamic_params']['omega']
        omega_dot = (omega - omega_before) / self._timer.dt

        hub_state_dict = state_dict['_hub']
        hub_state_dict = self._hub.modify_states(state_dict=hub_state_dict)
        state_dict['_hub'] = hub_state_dict

        for id in range(self._num_state_effectors):
            state_effector_id = f'_state_effector_{id}'
            state_effector: BaseStateEffector = self.get_submodule(
                state_effector_id)
            state_effector_state_dict: StateEffectorStateDict = state_dict.get(
                state_effector_id, {})
            state_effector_state_dict = state_effector.modify_states(
                state_dict=state_effector_state_dict,
                integrate_time_step=0. * self._timer.dt,
            )
            state_effector_state_dict = state_effector.calculate_force_torque_on_body(
                state_effector_state_dict,
                integrate_time_step=0. * self._timer.dt,
                omega_BN_B=omega,
            )
            state_dict[state_effector_id] = state_effector_state_dict

        state_dict = self.update_state_effectors(state_dict, *args, **kwargs)

        # prepare output
        position = state_dict['_hub']['dynamic_params']['pos']
        velocity = state_dict['_hub']['dynamic_params']['velocity']
        position_in_ineritial, velocity_in_inertial = self._gravity_field.update_inertial_position_and_velocity(
            position,
            velocity,
            central_body_ephemeris=central_body_ephemeris,
        )

        return state_dict, (SpacecraftStateOutput(
            position_in_inerital=position_in_ineritial,
            velocity_in_inertial=velocity_in_inertial,
            sigma=sigma,
            omega=omega,
            omega_dot=omega_dot,
            total_accumulated_non_gravitational_velocity_change_in_body=
            state_dict[
                'accumulated_non_gravitational_velocity_change_in_body'],
            total_accumulated_non_gravitational_velocity_change_in_inertial=
            state_dict[
                'accumulated_non_gravitational_velocity_change_in_inertial'],
            non_conservative_acceleration_of_body_in_body=
            non_conservative_acceleration_of_body_in_body,
        ), )
