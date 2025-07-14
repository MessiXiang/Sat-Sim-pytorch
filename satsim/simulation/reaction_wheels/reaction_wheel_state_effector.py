__all__ = ['ReactionWheelsStateDict', 'ReactionWheels']
from dataclasses import fields
from typing import Iterable, NotRequired, Sequence, TypedDict

import torch
import torch.nn.functional as F

from satsim.architecture import Module

from ..base import BackSubMatrices, StateEffectorMixin, StateEffectorStateDict
from .reaction_wheels import ReactionWheel


class ReactionWheelsDynamicParams(TypedDict):
    angular_velocity: torch.Tensor
    angular_acceleration: NotRequired[torch.Tensor]


class ReactionWheelsStateDict(StateEffectorStateDict):
    current_torque: torch.Tensor
    dynamic_params: ReactionWheelsDynamicParams


class ReactionWheels(
        Module[ReactionWheelsStateDict],
        StateEffectorMixin[ReactionWheelsStateDict],
):

    def __init__(
        self,
        *args,
        reaction_wheels: Iterable[ReactionWheel],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        reaction_wheels = list(reaction_wheels)
        if len(reaction_wheels) == 0:
            raise ValueError("Reaction Wheel list cannot be empty")

        states = {
            field.name:
            torch.tensor([
                getattr(reaction_wheel, field.name)
                for reaction_wheel in reaction_wheels
            ])
            for field in fields(ReactionWheel)
        }

        for key, value in states.items():
            dim = value.dim()
            if dim > 1:
                dims = list(range(value.dim()))
                new_dims = dims[1:] + dims[:1]
                value = value.permute(*new_dims)

            states[key] = value.unsqueeze(-2)

        self.angular_velocity_init = states.pop('angular_velocity_init')
        try:
            states['spin_axis_in_body'] = F.one_hot(
                states['spin_axis_in_body'].squeeze(-2),
                num_classes=3,
            ).transpose(-1, -2).float()
        except:
            breakpoint()

        for attr, value in states.items():
            self.register_buffer(attr, value, persistent=False)

    def reset(self) -> ReactionWheelsStateDict:
        state_dict = super().reset()

        state_effector_state_dict = super().state_effector_reset()
        state_dict.update(
            state_effector_state_dict,
            current_torque=torch.zeros_like(self.angular_velocity_init),
            dynamic_params=ReactionWheelsDynamicParams(
                angular_velocity=self.angular_velocity_init),
        )
        return state_dict

    def update_back_substitution_contribution(
        self,
        state_dict: ReactionWheelsStateDict,
        back_substitution_contribution: BackSubMatrices,
        sigma_BN: torch.Tensor,
        angular_velocity_BN_B: torch.Tensor,
        g_N: torch.Tensor,
    ) -> tuple[ReactionWheelsStateDict, BackSubMatrices]:
        angular_velocity = state_dict['dynamic_params']['angular_velocity']
        current_torque = state_dict['current_torque']
        moment_of_inertia_wrt_spin = self.get_buffer(
            'moment_of_inertia_wrt_spin')
        spin_axis_in_body = self.get_buffer('spin_axis_in_body')

        back_substitution_contribution[
            'matrix_d'] = back_substitution_contribution['matrix_d'] - (
                moment_of_inertia_wrt_spin *
                (spin_axis_in_body.unsqueeze(-2) *
                 spin_axis_in_body.unsqueeze(-3)).sum(-1))

        back_substitution_contribution[
            'vec_rot'] = back_substitution_contribution['vec_rot'] - (
                spin_axis_in_body * current_torque +
                moment_of_inertia_wrt_spin * angular_velocity * torch.cross(
                    angular_velocity_BN_B.unsqueeze(-1),
                    spin_axis_in_body,
                    dim=-2,
                )).sum(-1)

        return state_dict, back_substitution_contribution

    def compute_derivatives(
        self,
        state_dict: ReactionWheelsStateDict,
        rDDot_BN_N: torch.Tensor,
        angular_velocityDot_BN_B: torch.Tensor,
        sigma_BN: torch.Tensor,
    ) -> ReactionWheelsStateDict:
        dynamic_params = state_dict['dynamic_params']
        current_torque = state_dict['current_torque']
        moment_of_inertia_wrt_spin = self.get_buffer(
            'moment_of_inertia_wrt_spin')
        spin_axis_in_body = self.get_buffer('spin_axis_in_body')

        angular_acceleration = current_torque / moment_of_inertia_wrt_spin - torch.matmul(
            angular_velocityDot_BN_B.unsqueeze(-2),
            spin_axis_in_body,
        )
        dynamic_params['angular_acceleration'] = angular_acceleration

        state_dict['dynamic_params'] = dynamic_params
        return state_dict

    def update_energy_momentum_contributions(
        self,
        state_dict: ReactionWheelsStateDict,
        rotAngMomPntCContr_B: torch.Tensor,
        rotEnergyContr: torch.Tensor,
        angular_velocity_BN_B: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        moment_of_inertia_wrt_spin = self.get_buffer(
            'moment_of_inertia_wrt_spin')
        spin_axis_in_body = self.get_buffer('spin_axis_in_body')
        angular_velocity = angular_velocity = state_dict['dynamic_params'][
            'angular_velocity']

        rotAngMomPntCContr_B = (moment_of_inertia_wrt_spin *
                                spin_axis_in_body *
                                angular_velocity).sum(dim=-1)
        rotEnergyContr += (
            1. / 2 * moment_of_inertia_wrt_spin * angular_velocity**2 +
            moment_of_inertia_wrt_spin * angular_velocity * torch.matmul(
                angular_velocity_BN_B.unsqueeze(-2),
                spin_axis_in_body,
            )).sum()  # TODO: check dim

        return rotAngMomPntCContr_B, rotEnergyContr

    def forward(
        self,
        state_dict: ReactionWheelsStateDict,
        *args,
        motor_torque: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[
            ReactionWheelsStateDict,
            tuple,
    ]:
        """Processes the reaction wheel state and applies motor torque with constraints.

        This method updates the reaction wheel state dictionary by applying the provided motor torque,
        subject to constraints such as torque saturation, minimum torque threshold, power limits, and
        speed saturation. If no motor torque is provided, the method returns the unchanged state dictionary
        and a zero tensor. The method modifies the `reaction_wheels` dictionary within `state_dict` to
        store the applied torque and previous angular velocity, and updates the angular velocity based on
        dynamic parameters.

        Args:
            state_dict (ReactionWheelStateEffectorStateDict): Dictionary containing reaction wheel state,
                including 'reaction_wheels' with fields 'max_torque', 'min_torque', 'max_power_efficiency', 'angular_velocity',
                'max_angular_velocity', and 'dynamic_params' with 'angular_velocity'.
            *args: Variable length argument list (not used).
            motor_torque (torch.Tensor | None, optional): Motor torque to apply to the reaction wheels.
                If None, no torque is applied, and a zero tensor is returned in the output tuple.
                Defaults to None.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            tuple[ReactionWheelStateEffectorStateDict, tuple[torch.Tensor]]:
                - The updated `state_dict` with modified 'reaction_wheels' dictionary, including
                'current_torque' (applied torque) and
                updated 'angular_velocity' from dynamic parameters.
                - A tuple containing a single zero tensor with the same shape as 'angular_velocity', representing
                additional output (e.g., torque applied or residuals).

        Notes:
            - Torque constraints include:
            - **Torque saturation**: Clamps torque to [-max_torque, max_torque] where max_torque > 0.
            - **Minimum torque**: Sets torque to 0 if its absolute value is below min_torque.
            - **Power saturation**: Limits torque to max_power_efficiency / |angular_velocity| when power exceeds max_power_efficiency.
            - **Speed saturation**: Sets torque to 0 if |angular_velocity| >= max_angular_velocity and torque increases speed.
            - The method assumes 'angular_velocity' in 'dynamic_params' is precomputed for updating 'angular_velocity'.
        """
        max_torque = self.get_buffer('max_torque')
        max_power_efficiency = self.get_buffer('max_power_efficiency')
        angular_velocity = state_dict['dynamic_params']['angular_velocity']
        max_angular_velocity = self.get_buffer('max_angular_velocity')

        if max_torque.dim() != motor_torque.dim():
            motor_torque = motor_torque.unsqueeze(-2)

        torque_saturation_mask = max_torque > 0
        motor_torque = torch.where(
            torque_saturation_mask,
            torch.clamp(
                motor_torque,
                min=-max_torque,
                max=max_torque,
            ),
            motor_torque,
        )

        power_saturation_mask = (max_power_efficiency > 0) & \
            (torch.abs(motor_torque * angular_velocity) >=max_power_efficiency)
        motor_torque = torch.where(
            power_saturation_mask,
            torch.copysign(
                max_power_efficiency / angular_velocity,
                motor_torque,
            ),
            motor_torque,
        )

        speed_saturation_mask = (max_angular_velocity > 0.) & \
            (torch.abs(angular_velocity) >= max_angular_velocity ) & \
            (angular_velocity * motor_torque >= 0)
        motor_torque = torch.where(
            speed_saturation_mask,
            0.,
            motor_torque,
        )

        state_dict['current_torque'] = motor_torque

        return state_dict, tuple()
