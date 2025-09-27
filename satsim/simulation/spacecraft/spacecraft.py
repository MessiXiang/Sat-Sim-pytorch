__all__ = [
    'Spacecraft',
    'SpacecraftStateDict',
    'SpacecraftStateOutput',
    'DynamicParamsDict',
]
from copy import deepcopy
from typing import NamedTuple, NotRequired, TypedDict

import torch

from satsim.architecture import Module, Timer
from satsim.utils.matrix_support import mrp_to_rotation_matrix

from ..base import BackSubMatrices, MassProps
from ..gravity import GravityField
from ..reaction_wheels import (ReactionWheels, ReactionWheelsDynamicParams,
                               ReactionWheelsStateDict)
from .hub_effector import (HubEffector, HubEffectorDynamicParams,
                           HubEffectorStateDict)


class SpacecraftStateOutput(NamedTuple):
    position_BN_N: torch.Tensor
    velocity_BN_N: torch.Tensor
    angular_acceleration_BN_B: torch.Tensor
    total_accumulated_non_gravitational_velocity_change_BN_B: torch.Tensor
    total_accumulated_non_gravitational_velocity_change_BN_N: torch.Tensor
    non_conservative_acceleration_BN_B: torch.Tensor


class SpacecraftStateDict(TypedDict):
    ## Running created
    mass_props: MassProps
    accumulated_non_gravitational_velocity_change_BN_B: torch.Tensor
    accumulated_non_gravitational_velocity_change_BN_N: torch.Tensor

    ## childmodules
    _hub: HubEffectorStateDict
    _reaction_wheels: NotRequired[ReactionWheelsStateDict]


class DynamicParamsDict(TypedDict):
    _hub: HubEffectorDynamicParams
    _reaction_wheels: NotRequired[ReactionWheelsDynamicParams]


# MRPSwitchCount deserted
class Spacecraft(
        Module[SpacecraftStateDict], ):

    def __init__(
        self,
        *args,
        timer: Timer,
        mass: torch.Tensor,
        moment_of_inertia_matrix_wrt_body_point: torch.Tensor,
        position_BN_N: torch.Tensor,
        velocity_BN_N: torch.Tensor,
        attitude_BN: torch.Tensor | None = None,
        angular_velocity_BN_B: torch.Tensor | None = None,
        gravity_field: GravityField | None = None,
        reaction_wheels: ReactionWheels | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, timer=timer, **kwargs)

        self._hub = HubEffector(
            timer=timer,
            mass=mass,
            moment_of_inertia_matrix_wrt_body_point=
            moment_of_inertia_matrix_wrt_body_point,
            position=position_BN_N,
            velocity=velocity_BN_N,
            attitude=attitude_BN,
            angular_velocity=angular_velocity_BN_B,
        )
        self._gravity_field = gravity_field
        self._reaction_wheels = reaction_wheels

    @property
    def gravity_field(self) -> GravityField | None:
        return self._gravity_field

    @property
    def reaction_wheels(self) -> ReactionWheels | None:
        return self._reaction_wheels

    def reset(self) -> SpacecraftStateDict:
        state_dict = super().reset()
        state_dict = self.update_spacecraft_mass_props(
            state_dict,
            0. * self._timer.dt,
        )
        state_dict.update(
            accumulated_non_gravitational_velocity_change_BN_B=torch.zeros(3),
            accumulated_non_gravitational_velocity_change_BN_N=torch.zeros(3),
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
        mass = torch.tensor(0.)
        moment_of_inertia_matrix_wrt_body_point = torch.tensor(0.)

        if self._reaction_wheels is not None:
            reaction_wheels_state_dict = state_dict.get('_reaction_wheels')
            spacecraft_mass_props = self._reaction_wheels.update_effector_mass(
                state_dict=reaction_wheels_state_dict,
                integrate_time_step=integrate_time_step,
            )
            mass = mass + spacecraft_mass_props['mass']
            moment_of_inertia_matrix_wrt_body_point = moment_of_inertia_matrix_wrt_body_point + spacecraft_mass_props[
                'moment_of_inertia_matrix_wrt_body_point']

        mass = mass + self._hub.mass
        moment_of_inertia_matrix_wrt_body_point = (
            moment_of_inertia_matrix_wrt_body_point +
            self._hub.moment_of_inertia_matrix_wrt_body_point)

        state_dict['mass_props'] = MassProps(
            mass=mass,
            moment_of_inertia_matrix_wrt_body_point=
            moment_of_inertia_matrix_wrt_body_point,
        )
        return state_dict

    def equation_of_motion(
        self,
        state_dict: SpacecraftStateDict,
        integrate_time_step: float,
    ) -> DynamicParamsDict:
        state_dict = self.update_spacecraft_mass_props(
            state_dict,
            integrate_time_step=integrate_time_step,
        )
        mass = state_dict['mass_props']['mass']
        moment_of_inertia_matrix_wrt_body_point = state_dict['mass_props'][
            'moment_of_inertia_matrix_wrt_body_point']

        hub_state_dict = state_dict['_hub']
        position_BN_N = hub_state_dict['dynamic_params']['position_BN_N']
        angular_velocity_BN_B = hub_state_dict['dynamic_params'][
            'angular_velocity_BN_B']

        gravity_acceleration_B_N: torch.Tensor
        if self.gravity_field is not None:
            _, (gravity_acceleration_B_N, ) = self.gravity_field(
                position_BN_N, )
        else:
            gravity_acceleration_B_N = torch.zeros_like(position_BN_N)

        # calculate back substitution matrices
        back_substitution_contribution = BackSubMatrices(
            moment_of_inertia_matrix=torch.zeros_like(
                moment_of_inertia_matrix_wrt_body_point),
            ext_force=torch.zeros_like(position_BN_N),
            ext_torque=torch.zeros_like(position_BN_N),
        )

        # Update back substitution matrices for state effectors
        # Update reactionwheels
        if self._reaction_wheels is not None:
            reaction_wheels_state_dict = state_dict.get('_reaction_wheels')
            back_substitution_contribution = self._reaction_wheels.update_back_substitution_contribution(
                state_dict=reaction_wheels_state_dict,
                back_substitution_contribution=back_substitution_contribution,
                angular_velocity_BN_B=angular_velocity_BN_B,
            )

        # update back substitution matrices for hub
        back_substitution_contribution['moment_of_inertia_matrix'] = (
            back_substitution_contribution['moment_of_inertia_matrix'] +
            moment_of_inertia_matrix_wrt_body_point)
        back_substitution_contribution['ext_torque'] = (
            back_substitution_contribution['ext_torque'] + torch.cross(
                torch.einsum(
                    '...ij, ...j -> ...i',
                    moment_of_inertia_matrix_wrt_body_point,
                    angular_velocity_BN_B,
                ),
                angular_velocity_BN_B,
                dim=-1,
            ))

        # We now exclude gravitational force as ext_force, but add it in
        # velocity's derivative in the end in  compute_derivatives method
        # gravity_force_in_inertial = gravity_acceleration * mass
        # gravity_force_in_body = torch.matmul(
        #     direction_cosine_matrix_body_to_inertial.transpose(-1, -2),
        #     gravity_force_in_inertial.unsqueeze(-1),
        # ).squeeze(-1)
        # back_substitution_contribution['ext_force'] = back_substitution_contribution['ext_force'] + \
        #     gravity_force_in_body

        hub_state_dot = self._hub.compute_derivatives(
            state_dict=hub_state_dict,
            integrate_time_step=integrate_time_step,
            back_substitution_matrices=back_substitution_contribution,
            gravity_acceleration=gravity_acceleration_B_N,
            spacecraft_mass=mass,
        )

        states_dot = DynamicParamsDict(_hub=hub_state_dot)
        if self._reaction_wheels is not None:
            reaction_wheels_state_dict = state_dict.get('_reaction_wheels')
            reaction_wheels_state_dot = self._reaction_wheels.compute_derivatives(
                state_dict=reaction_wheels_state_dict,
                integrate_time_step=integrate_time_step,
                angular_velocity_dot=hub_state_dot['angular_velocity'],
            )
            states_dot['_reaction_wheels'] = reaction_wheels_state_dot

        return states_dot

    def get_dynamic_params(
        self,
        state_dict: SpacecraftStateDict,
    ) -> DynamicParamsDict:
        dynamic_params = dict()

        hub_state_dict = state_dict['_hub']
        dynamic_params['_hub'] = hub_state_dict['dynamic_params']

        if self._reaction_wheels is not None:
            reaction_wheels_state_dict = state_dict.get('_reaction_wheels')
            dynamic_params['_reaction_wheels'] = (
                reaction_wheels_state_dict['dynamic_params'])

        return dynamic_params

    def apply_dynamic_params(
        self,
        state_dict: SpacecraftStateDict,
        dynamic_params: DynamicParamsDict,
    ) -> SpacecraftStateDict:
        hub_state: HubEffectorDynamicParams = dynamic_params['_hub']
        state_dict['_hub']['dynamic_params'] = hub_state

        if self._reaction_wheels is not None:
            state = dynamic_params.get('_reaction_wheels')
            reaction_wheels_state_dict = state_dict.get('_reaction_wheels')
            reaction_wheels_state_dict['dynamic_params'] = state
            state_dict.update(_reaction_wheels=reaction_wheels_state_dict)

        return state_dict

    def integrate_to_this_time(
        self,
        state_dict: SpacecraftStateDict,
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
        )

        # stage 2
        for module_name, module_dynamic_params in dynamic_params.items():
            module_dynamic_params_dot = k1[module_name]
            module_previous_dynamic_state = previous_dynamic_params[
                module_name]
            for state_name in module_dynamic_params.keys():
                module_dynamic_params[state_name] = (
                    module_previous_dynamic_state[state_name] + 0.5 *
                    self._timer.dt * module_dynamic_params_dot[state_name])
        state_dict = self.apply_dynamic_params(state_dict, dynamic_params)
        k2 = self.equation_of_motion(
            state_dict=state_dict,
            integrate_time_step=-0.5 * self._timer.dt,
        )

        # state 3
        for module_name, module_dynamic_params in dynamic_params.items():
            module_dynamic_params_dot = k2[module_name]
            module_previous_dynamic_state = previous_dynamic_params[
                module_name]
            for state_name in module_dynamic_params.keys():
                module_dynamic_params[state_name] = (
                    module_previous_dynamic_state[state_name] + 0.5 *
                    self._timer.dt * module_dynamic_params_dot[state_name])
        state_dict = self.apply_dynamic_params(state_dict, dynamic_params)
        k3 = self.equation_of_motion(
            state_dict=state_dict,
            integrate_time_step=-0.5 * self._timer.dt,
        )

        # stage 4
        for module_name, module_dynamic_params in dynamic_params.items():
            module_dynamic_params_dot = k3[module_name]
            module_previous_dynamic_state = previous_dynamic_params[
                module_name]
            for state_name in module_dynamic_params.keys():
                module_dynamic_params[state_name] = (
                    module_previous_dynamic_state[state_name] +
                    1 * self._timer.dt * module_dynamic_params_dot[state_name])
        state_dict = self.apply_dynamic_params(state_dict, dynamic_params)
        k4 = self.equation_of_motion(
            state_dict=state_dict,
            integrate_time_step=0. * self._timer.dt,
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
                module_dynamic_params[state_name] = (
                    module_previous_dynamic_state[state_name] +
                    1 / 6 * self._timer.dt * module_state_dot_k1[state_name] +
                    1 / 3 * self._timer.dt * module_state_dot_k2[state_name] +
                    1 / 3 * self._timer.dt * module_state_dot_k3[state_name] +
                    1 / 6 * self._timer.dt * module_state_dot_k4[state_name])
        state_dict = self.apply_dynamic_params(state_dict, dynamic_params)

        return state_dict

    def forward(
        self,
        state_dict: SpacecraftStateDict,
        *args,
        **kwargs,
    ) -> tuple[SpacecraftStateDict, SpacecraftStateOutput]:
        ## pre-solution
        hub_state_dict = state_dict['_hub']
        # for non gravitational acceleration calculation
        # hub_state_dict = self._hub.match_gravity_to_velocity_state(
        #     hub_state_dict)
        # state_dict['_hub'] = hub_state_dict
        angular_velocity_BN_B_before = hub_state_dict['dynamic_params'][
            'angular_velocity'].clone()

        ## solve next state
        state_dict = self.integrate_to_this_time(state_dict=state_dict)

        ## post-solution
        # update mass props
        # state_dict = self.update_spacecraft_mass_props(
        #     state_dict,
        #     integrate_time_step=0. * self._timer.dt,
        # )
        velocity_BN_N = state_dict['_hub']['dynamic_params']['velocity']
        attitude_BN = state_dict['_hub']['dynamic_params']['attitude']
        direction_cosine_matrix_BN = mrp_to_rotation_matrix(attitude_BN)
        gravitational_velocity_BN_N = state_dict['_hub']['dynamic_params'][
            'grav_velocity']
        state_dict[
            'accumulated_non_gravitational_velocity_change_BN_B'] = state_dict[
                'accumulated_non_gravitational_velocity_change_BN_B'] + torch.einsum(
                    '...ij, ...j -> ...i',
                    direction_cosine_matrix_BN,
                    velocity_BN_N - gravitational_velocity_BN_N,
                )
        state_dict['accumulated_non_gravitational_velocity_change_BN_N'] = (
            state_dict['accumulated_non_gravitational_velocity_change_BN_N'] +
            (velocity_BN_N - gravitational_velocity_BN_N))

        non_conservative_acceleration_BN_B = torch.einsum(
            '...ij, ...j -> ...i',
            direction_cosine_matrix_BN,
            (velocity_BN_N - gravitational_velocity_BN_N),
        ) / self._timer.dt

        angular_velocity_BN_B = state_dict['_hub']['dynamic_params'][
            'angular_velocity']
        angular_acceleration = (angular_velocity_BN_B -
                                angular_velocity_BN_B_before) / self._timer.dt

        hub_state_dict = state_dict['_hub']
        hub_state_dict = self._hub.normalize_attitude(
            state_dict=hub_state_dict, )
        state_dict['_hub'] = hub_state_dict

        # NOTE: Here originally calculate ext_force_torque on spacecraft.

        # prepare output
        position_BN_N = state_dict['_hub']['dynamic_params']['position']
        velocity_BN_N = state_dict['_hub']['dynamic_params']['velocity']
        if self.gravity_field is not None:
            position_BN_N, velocity_BN_N = self.gravity_field.update_inertial_position_and_velocity(
                position_BN_N,
                velocity_BN_N,
            )
        else:
            position_BN_N = position_BN_N
            velocity_BN_N = velocity_BN_N

        return state_dict, SpacecraftStateOutput(
            position_BN_N=position_BN_N,
            velocity_BN_N=velocity_BN_N,
            angular_acceleration_BN_B=angular_acceleration,
            total_accumulated_non_gravitational_velocity_change_BN_B=state_dict[
                'accumulated_non_gravitational_velocity_change_BN_B'],
            total_accumulated_non_gravitational_velocity_change_BN_N=state_dict[
                'accumulated_non_gravitational_velocity_change_BN_N'],
            non_conservative_acceleration_BN_B=
            non_conservative_acceleration_BN_B,
        )
