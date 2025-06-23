__all__ = ['ReactionWheelStateEffectorStateDict', 'ReactionWheelStateEffector']
from math import sqrt

import torch

from satsim.architecture import Module

from ..base import BackSubMatrices, StateEffectorMixin, StateEffectorStateDict
from .data import ReactionWheelDynamicParams, ReactionWheelsOutput
from .reaction_wheels import ReactionWheels


class ReactionWheelStateEffectorStateDict(StateEffectorStateDict):
    reaction_wheels: ReactionWheels
    dynamic_params: ReactionWheelDynamicParams


class ReactionWheelStateEffector(
        Module[ReactionWheelStateEffectorStateDict],
        StateEffectorMixin[ReactionWheelStateEffectorStateDict],
):

    def reset(self) -> ReactionWheelStateEffectorStateDict:
        state_dict = super().reset()

        state_effector_state_dict = super().state_effector_reset()
        state_dict.update(
            state_effector_state_dict,
            reaction_wheels=ReactionWheels(),
        )
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

        assert reaction_wheels.is_jitter.sum().item() == 0
        dynamic_params['omega'] = reaction_wheels.Omega

        state_dict.update(dynamic_params=dynamic_params)

        return state_dict

    def update_effector_mass(
        self,
        state_dict: ReactionWheelStateEffectorStateDict,
    ) -> ReactionWheels:
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

        friction_stribeck_mask = torch.abs(
            reaction_wheels.Omega
        ) < 0.1 * reaction_wheels.omegaLimitCycle and reaction_wheels.beta_static > 0
        delta_omega = reaction_wheels.Omega - reaction_wheels.omega_before
        sign_of_omega = torch.sign(reaction_wheels.Omega)
        sign_of_delta_omega = torch.sign(delta_omega)
        friction_stribeck = reaction_wheels.friction_stribeck
        reaction_wheels.friction_stribeck = friction_stribeck or \
            friction_stribeck_mask and \
            sign_of_delta_omega == sign_of_omega and \
            reaction_wheels.beta_static > 0

        omega_over_betastatic = reaction_wheels.Omega / reaction_wheels.beta_static
        omega_limit_cycle_over_betastatic = reaction_wheels.omegaLimitCycle / reaction_wheels.beta_static
        friction_force = torch.where(
            reaction_wheels.friction_stribeck,
            sqrt(2.0 * torch.e) * (reaction_wheels.friction_static -
                                   reaction_wheels.friction_coulomb) *
            torch.exp(-(omega_over_betastatic)**2 / 2.) *
            omega_over_betastatic / sqrt(2.) +
            reaction_wheels.friction_coulomb *
            torch.tanh(omega_over_betastatic * 10.) +
            reaction_wheels.cViscous * reaction_wheels.Omega,
            sign_of_omega * reaction_wheels.friction_coulomb +
            reaction_wheels.cViscous * reaction_wheels.Omega,
        )
        friction_force_at_limit_cycle = torch.where(
            reaction_wheels.friction_stribeck,
            sqrt(2.0 * torch.e) * (reaction_wheels.friction_static -
                                   reaction_wheels.friction_coulomb) *
            torch.exp(-(omega_limit_cycle_over_betastatic)**2 / 2.) *
            omega_limit_cycle_over_betastatic / sqrt(2.) +
            reaction_wheels.friction_coulomb *
            torch.tanh(omega_limit_cycle_over_betastatic * 10.) +
            reaction_wheels.cViscous * reaction_wheels.omegaLimitCycle,
            reaction_wheels.friction_coulomb +
            reaction_wheels.cViscous * reaction_wheels.omegaLimitCycle,
        )

        avoid_limit_cycle_friction_mask = torch.abs(
            reaction_wheels.Omega) < reaction_wheels.omegaLimitCycle
        friction_force = torch.where(
            avoid_limit_cycle_friction_mask, friction_force_at_limit_cycle /
            reaction_wheels.omegaLimitCycle * reaction_wheels.Omega,
            friction_force)

        reaction_wheels.friction_torque = -friction_force

        back_substitution_contribution[
            'matrix_d'] = back_substitution_contribution[
                'matrix_d'] - reaction_wheels.Js * torch.einsum(
                    "a b n, b c n -> a c n",
                    reaction_wheels.gsHat_B.unsqueeze(1),
                    reaction_wheels.gsHat_B.unsqueeze(0),
                )
        back_substitution_contribution[
            'vec_rot'] = back_substitution_contribution[
                'vec_rot'] - reaction_wheels.gsHat_B * (
                    reaction_wheels.current_torque +
                    reaction_wheels.friction_torque
                ) + reaction_wheels.Js * reaction_wheels.Omega * torch.cross(
                    omega_BN_B.unsqueeze(1), reaction_wheels.gsHat_B, dim=0)

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

        delta_omega = (reaction_wheels.current_torque + reaction_wheels.
                       friction_torque) / reaction_wheels.Js - torch.einsum(
                           "a b, a -> b",
                           reaction_wheels.gsHat_B,
                           omegaDot_BN_B,
                       ).unsqueeze(0)
        dynamic_params['delta_omega'] = delta_omega
        return state_dict

    def update_energy_momentum_contributions(
        self,
        state_dict: ReactionWheelStateEffectorStateDict,
        rotAngMomPntCContr_B: torch.Tensor,
        rotEnergyContr: float,
        omega_BN_B: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        reaction_wheels = state_dict['reaction_wheels']
        rotAngMomPntCContr_B = (reaction_wheels.Js * reaction_wheels.gsHat_B *
                                reaction_wheels.Omega).sum(dim=-1)
        rotEnergyContr += (
            1. / 2 * reaction_wheels.Js * reaction_wheels.Omega**2 +
            reaction_wheels.Js * reaction_wheels.Omega * torch.einsum(
                "a b, a -> b",
                reaction_wheels.gsHat_B,
                omega_BN_B,
            )).sum().item()

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

        reaction_wheels.Omega = dynamic_params['omega']
        return state_dict, (reaction_wheels.Omega)

    def forward(
        self,
        state_dict: ReactionWheelStateEffectorStateDict,
        *args,
        motor_torque: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[
            ReactionWheelStateEffectorStateDict,
            tuple[ReactionWheelsOutput],
    ]:
        reaction_wheels = state_dict['reaction_wheels']

        if motor_torque is None:
            return state_dict, tuple(reaction_wheels.export())

        torque_saturation_mask = reaction_wheels.max_torque > 0
        u_max = reaction_wheels.max_torque
        motor_torque = torch.where(
            torque_saturation_mask,
            torch.clamp(
                motor_torque,
                min=-u_max,
                max=u_max,
            ),
            motor_torque,
        )

        torque_ignore_mask = torch.abs(
            motor_torque) < reaction_wheels.min_torque
        motor_torque = torch.where(
            torque_ignore_mask,
            0.,
            motor_torque,
        )

        power_saturation_mask = reaction_wheels.max_power > 0 and \
            torch.abs(motor_torque * reaction_wheels.Omega) >= reaction_wheels.max_power
        motor_torque = torch.where(
            power_saturation_mask,
            torch.copysign(
                reaction_wheels.max_power / reaction_wheels.Omega,
                motor_torque,
            ),
            motor_torque,
        )

        speed_saturation_mask = torch.abs(reaction_wheels.Omega) >= reaction_wheels.max_omega and \
            reaction_wheels.max_omega > 0. and \
            reaction_wheels.Omega * motor_torque >= 0.
        motor_torque = torch.where(
            speed_saturation_mask,
            0.,
            motor_torque,
        )

        reaction_wheels.current_torque = motor_torque
        reaction_wheels.omega_before = reaction_wheels.Omega

        # Preparing output
        reaction_wheels.Omega = state_dict['dynamic_params']['omega']

        return state_dict, tuple(reaction_wheels.export())
