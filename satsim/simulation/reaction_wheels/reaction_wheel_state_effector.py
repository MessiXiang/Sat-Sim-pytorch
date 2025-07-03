__all__ = ['ReactionWheelStateEffectorStateDict', 'ReactionWheelStateEffector']
from math import sqrt

import torch

from satsim.architecture import Module

from ..base import BackSubMatrices, StateEffectorMixin, StateEffectorStateDict
from .data import ReactionWheelDynamicParams
from .reaction_wheels import ReactionWheelsStateDict


class ReactionWheelStateEffectorStateDict(StateEffectorStateDict):
    reaction_wheels: ReactionWheelsStateDict
    dynamic_params: ReactionWheelDynamicParams


class ReactionWheelStateEffector(
        Module[ReactionWheelStateEffectorStateDict],
        StateEffectorMixin[ReactionWheelStateEffectorStateDict],
):

    def reset(self) -> ReactionWheelStateEffectorStateDict:
        state_dict = super().reset()

        state_effector_state_dict = super().state_effector_reset()
        state_dict.update(state_effector_state_dict)
        return state_dict

    def link_in_states(self, dynamic_params: ReactionWheelDynamicParams):
        # Currently saved for code comprehension.
        # This method read vehicle gravity from dynamic parmas and storage in self.g_N
        pass

    def register_states(
        self,
        state_dict: ReactionWheelStateEffectorStateDict,
    ) -> ReactionWheelStateEffectorStateDict:
        dynamic_params: ReactionWheelDynamicParams = dict()
        reaction_wheels = state_dict['reaction_wheels']

        dynamic_params['omega'] = reaction_wheels.omega

        state_dict.update(dynamic_params=dynamic_params)

        return state_dict

    def update_effector_mass(
        self,
        state_dict: ReactionWheelStateEffectorStateDict,
    ) -> ReactionWheelStateEffectorStateDict:
        effProps = dict(
            mEff=0.,
            mEffDot=0.,
            IEffPntB_B=torch.zeros(3, 3),
            rEff_CB_B=torch.zeros(3),
            rEffPrime_CB_B=torch.zeros(3),
            IEffPrimePntB_B=torch.zeros(3, 3),
        )

        state_dict['effProps'] = effProps
        return state_dict

    def update_back_substitution_contribution(
        self,
        state_dict: ReactionWheelStateEffectorStateDict,
        back_substitution_contribution: BackSubMatrices,
        sigma_BN: torch.Tensor,
        omega_BN_B: torch.Tensor,
        g_N: torch.Tensor,
    ) -> tuple[ReactionWheelStateEffectorStateDict, BackSubMatrices]:
        reaction_wheels = state_dict['reaction_wheels']
        omega = reaction_wheels['omega']
        omega_limit_cycle = reaction_wheels['omegaLimitCycle']
        beta_static = reaction_wheels['beta_static']
        omega_before = reaction_wheels['omega_before']
        friction_stribeck = reaction_wheels['friction_stribeck']
        friction_static = reaction_wheels['friction_static']
        friction_coulomb = reaction_wheels['friction_coulomb']
        cViscous = reaction_wheels['cViscous']
        Js = reaction_wheels['Js']
        gsHat_B = reaction_wheels['gsHat_B']
        current_torque = reaction_wheels['current_torque']

        friction_stribeck_mask = torch.abs(
            omega) < 0.1 * omega_limit_cycle and beta_static > 0
        delta_omega = omega - omega_before
        sign_of_omega = torch.sign(omega)
        sign_of_delta_omega = torch.sign(delta_omega)
        reaction_wheels['friction_stribeck'] = friction_stribeck or \
            friction_stribeck_mask and \
            sign_of_delta_omega == sign_of_omega and \
            beta_static > 0

        omega_over_betastatic = omega / beta_static
        omega_limit_cycle_over_betastatic = omega_limit_cycle / beta_static
        friction_force = torch.where(
            friction_stribeck,
            sqrt(2.0 * torch.e) * (friction_static - friction_coulomb) *
            torch.exp(-(omega_over_betastatic)**2 / 2.) *
            omega_over_betastatic / sqrt(2.) +
            friction_coulomb * torch.tanh(omega_over_betastatic * 10.) +
            cViscous * omega,
            sign_of_omega * friction_coulomb + cViscous * omega,
        )
        friction_force_at_limit_cycle = torch.where(
            friction_stribeck,
            sqrt(2.0 * torch.e) * (friction_static - friction_coulomb) *
            torch.exp(-(omega_limit_cycle_over_betastatic)**2 / 2.) *
            omega_limit_cycle_over_betastatic / sqrt(2.) + friction_coulomb *
            torch.tanh(omega_limit_cycle_over_betastatic * 10.) +
            cViscous * omega_limit_cycle,
            friction_coulomb + cViscous * omega_limit_cycle,
        )

        avoid_limit_cycle_friction_mask = torch.abs(omega) < omega_limit_cycle
        friction_force = torch.where(
            avoid_limit_cycle_friction_mask,
            friction_force_at_limit_cycle / omega_limit_cycle * omega,
            friction_force)

        friction_torque = -friction_force
        reaction_wheels['friction_torque'] = friction_torque

        back_substitution_contribution[
            'matrix_d'] = back_substitution_contribution['matrix_d'] - Js * (
                gsHat_B.unsqueeze(1) * gsHat_B.unsqueeze(0))
        back_substitution_contribution[
            'vec_rot'] = back_substitution_contribution['vec_rot'] - gsHat_B * (
                current_torque + friction_torque) + Js * omega * torch.cross(
                    omega_BN_B.unsqueeze(1), gsHat_B, dim=0)

        return state_dict, back_substitution_contribution

    def compute_derivatives(
        self,
        state_dict: ReactionWheelStateEffectorStateDict,
        rDDot_BN_N: torch.Tensor,
        omegaDot_BN_B: torch.Tensor,
        sigma_BN: torch.Tensor,
    ) -> ReactionWheelStateEffectorStateDict:
        reaction_wheels = state_dict['reaction_wheels']
        dynamic_params = state_dict['dynamic_params']

        current_torque = reaction_wheels['current_torque']
        friction_torque = reaction_wheels['friction_torque']
        Js = reaction_wheels['Js']
        gsHat_B = reaction_wheels['gsHat_B']

        delta_omega = (current_torque + friction_torque) / Js - (
            gsHat_B.t() @ omegaDot_BN_B).unsqueeze(0)
        dynamic_params['delta_omega'] = delta_omega
        return state_dict

    def update_energy_momentum_contributions(
        self,
        state_dict: ReactionWheelStateEffectorStateDict,
        rotAngMomPntCContr_B: torch.Tensor,
        rotEnergyContr: torch.Tensor,
        omega_BN_B: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reaction_wheels = state_dict['reaction_wheels']
        Js = reaction_wheels['Js']
        gsHat_B = reaction_wheels['gsHat_B']
        omega = reaction_wheels['omega']

        rotAngMomPntCContr_B = (Js * gsHat_B * omega).sum(dim=-1)
        rotEnergyContr += (1. / 2 * Js * omega**2 + Js * omega *
                           (gsHat_B.t() @ omega_BN_B)).sum()

        return rotAngMomPntCContr_B, rotEnergyContr

    def get_state_output(
        self,
        state_dict: ReactionWheelStateEffectorStateDict,
    ) -> tuple[
            ReactionWheelStateEffectorStateDict,
            tuple[torch.Tensor],
    ]:
        dynamic_params = state_dict['dynamic_params']
        reaction_wheels = state_dict['reaction_wheels']

        reaction_wheels['omega'] = dynamic_params['omega']
        return state_dict, (reaction_wheels['omega'], )

    def forward(
        self,
        state_dict: ReactionWheelStateEffectorStateDict,
        *args,
        motor_torque: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[
            ReactionWheelStateEffectorStateDict,
            tuple[torch.Tensor],
    ]:
        reaction_wheels = state_dict['reaction_wheels']
        max_torque = reaction_wheels['max_torque']
        min_torque = reaction_wheels['min_torque']
        max_power = reaction_wheels['max_power']
        omega = reaction_wheels['omega']
        max_omega = reaction_wheels['max_omega']

        if motor_torque is None:
            return state_dict, (torch.zeros_like(omega), )

        torque_saturation_mask = max_torque > 0
        u_max = max_torque
        motor_torque = torch.where(
            torque_saturation_mask,
            torch.clamp(
                motor_torque,
                min=-u_max,
                max=u_max,
            ),
            motor_torque,
        )

        torque_ignore_mask = torch.abs(motor_torque) < min_torque
        motor_torque = torch.where(
            torque_ignore_mask,
            0.,
            motor_torque,
        )

        power_saturation_mask = max_power > 0 and \
            torch.abs(motor_torque * omega) >=max_power
        motor_torque = torch.where(
            power_saturation_mask,
            torch.copysign(
                max_power / omega,
                motor_torque,
            ),
            motor_torque,
        )

        speed_saturation_mask = torch.abs(omega) >= max_omega and \
            max_omega > 0. and \
            omega * motor_torque >= 0
        motor_torque = torch.where(
            speed_saturation_mask,
            0.,
            motor_torque,
        )

        reaction_wheels['current_torque'] = motor_torque
        reaction_wheels['omega_before'] = omega

        # Preparing output
        reaction_wheels['omega'] = state_dict['dynamic_params']['omega']

        return state_dict, (torch.zeros_like(omega), )
