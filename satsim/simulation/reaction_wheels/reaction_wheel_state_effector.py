__all__ = [
    'ReactionWheelsStateDict',
    'ReactionWheels',
    'ReactionWheelsDynamicParams',
]
from dataclasses import fields
from typing import Iterable, TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor

from ..base import (BackSubMatrices, BaseStateEffector, BatteryStateDict,
                    PowerNodeMixin, StateEffectorStateDict)
from .reaction_wheels import ReactionWheel


class ReactionWheelsDynamicParams(TypedDict):
    angular_velocity: Tensor


class ReactionWheelsStateDict(
        StateEffectorStateDict[ReactionWheelsDynamicParams]):
    current_torque: Tensor
    power_status: Tensor


class ReactionWheels(
        BaseStateEffector[ReactionWheelsStateDict],
        PowerNodeMixin,
):

    def __init__(
        self,
        *args,
        reaction_wheels: Iterable[ReactionWheel],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        reaction_wheels = list(reaction_wheels)
        if len(reaction_wheels) != 3:
            raise ValueError(
                "Reaction Wheel list must contain three reaction_wheels")

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

        self.register_buffer(
            '_moment_of_inertia_wrt_spin',
            states['moment_of_inertia_wrt_spin'],
            persistent=False,
        )
        self.register_buffer(
            '_max_torque',
            states['max_torque'],
            persistent=False,
        )
        self.register_buffer(
            '_mass',
            states['mass'],
            persistent=False,
        )
        self.register_buffer(
            '_max_angular_velocity',
            states['max_angular_velocity'],
            persistent=False,
        )
        self.register_buffer(
            '_max_power_efficiency',
            states['max_power_efficiency'],
            persistent=False,
        )
        self.register_buffer(
            '_base_power',
            states['base_power'],
            persistent=False,
        )
        self.register_buffer(
            '_elec_to_mech_efficiency',
            states['elec_to_mech_efficiency'],
            persistent=False,
        )
        self.register_buffer(
            '_mech_to_elec_efficiency',
            states['mech_to_elec_efficiency'],
            persistent=False,
        )

    @property
    def spin_axis_in_body(self) -> Tensor:
        return torch.eye(3, 3).expand(self.mass.size(0), 3, 3).to(self.mass)

    @property
    def moment_of_inertia_wrt_spin(self) -> Tensor:
        return self.get_buffer('_moment_of_inertia_wrt_spin')

    @property
    def max_torque(self) -> Tensor:
        return self.get_buffer('_max_torque')

    @property
    def mass(self) -> Tensor:
        return self.get_buffer('_mass')

    @property
    def max_angular_velocity(self) -> Tensor:
        return self.get_buffer('_max_angular_velocity')

    @property
    def max_power_efficiency(self) -> Tensor:
        return self.get_buffer('_max_power_efficiency')

    @property
    def base_power(self) -> Tensor:
        return self.get_buffer('_base_power')

    @property
    def elec_to_mech_efficiency(self) -> Tensor:
        return self.get_buffer('_elec_to_mech_efficiency')

    @property
    def mech_to_elec_efficiency(self) -> Tensor:
        return self.get_buffer('_mech_to_elec_efficiency')

    def reset(self) -> ReactionWheelsStateDict:
        state_dict = super().reset()
        state_dict.update(
            current_torque=torch.zeros_like(self.angular_velocity_init),
            dynamic_params=ReactionWheelsDynamicParams(
                angular_velocity=self.angular_velocity_init),
        )
        return state_dict

    def update_back_substitution_contribution(
        self,
        state_dict: ReactionWheelsStateDict,
        back_substitution_contribution: BackSubMatrices,
        angular_velocity_BN_B: Tensor,
    ) -> BackSubMatrices:
        angular_velocity = state_dict['dynamic_params']['angular_velocity']
        current_torque = state_dict['current_torque']

        back_substitution_contribution[
            'moment_of_inertia_matrix'] = back_substitution_contribution[
                'moment_of_inertia_matrix'] - (
                    self.moment_of_inertia_wrt_spin * torch.einsum(
                        '... a n, ... b n -> ... a b',
                        self.spin_axis_in_body,
                        self.spin_axis_in_body,
                    ))

        back_substitution_contribution[
            'ext_torque'] = back_substitution_contribution['ext_torque'] - (
                self.spin_axis_in_body * current_torque +
                self.moment_of_inertia_wrt_spin * angular_velocity *
                torch.cross(
                    angular_velocity_BN_B.unsqueeze(-1),
                    self.spin_axis_in_body,
                    dim=-2,
                )).sum(-1)

        return back_substitution_contribution

    def compute_derivatives(
        self,
        state_dict: ReactionWheelsStateDict,
        integrate_time_step: float,
        angular_velocity_dot: Tensor,
    ) -> ReactionWheelsDynamicParams:
        dynamic_params = state_dict['dynamic_params']
        current_torque = state_dict['current_torque']

        angular_acceleration = (
            current_torque / self.moment_of_inertia_wrt_spin - torch.matmul(
                angular_velocity_dot.unsqueeze(-2),
                self.spin_axis_in_body,
            ))

        dynamic_params['angular_velocity'] = angular_acceleration
        return dynamic_params

    def compute_power_usage(
        self,
        battery_state_dict: BatteryStateDict,
        state_dict: ReactionWheelsStateDict,
    ) -> tuple[BatteryStateDict, ReactionWheelsStateDict]:
        angular_velocity = state_dict['dynamic_params']['angular_velocity']
        current_torque = state_dict['current_torque']
        wheel_power = angular_velocity * current_torque  # shape (batch_size, 1,num_reaction_wheels)

        is_accel_mode = (self.mech_to_elec_efficiency < 0) | (
            wheel_power > 0)  # shape (batch_size, 1, num_reaction_wheels)

        accel_power = self.base_power + torch.abs(
            wheel_power
        ) / self.elec_to_mech_efficiency  # shape (batch_size, 1, num_reaction_wheels)
        regen_power = self.base_power + self.mech_to_elec_efficiency * wheel_power  # shape (batch_size, 1, num_reaction_wheels)

        total_power = torch.where(
            is_accel_mode,
            accel_power,
            regen_power,
        ).sum(-1).squeeze(-1)  # shape (n_s)

        power_status, battery_state_dict = self.update_power_status(
            -total_power,
            battery_state_dict,
        )

        state_dict['current_torque'] = torch.where(
            power_status.unsqueeze(-1).unsqueeze(-1),
            current_torque,
            0.,
        )

        return battery_state_dict, state_dict

    def forward(
        self,
        state_dict: ReactionWheelsStateDict,
        *args,
        battery_state_dict: BatteryStateDict,
        motor_torque: Tensor,
        **kwargs,
    ) -> tuple[
            ReactionWheelsStateDict,
            tuple[BatteryStateDict],
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
            motor_torque (Tensor | None, optional): Motor torque to apply to the reaction wheels.
                If None, no torque is applied, and a zero tensor is returned in the output tuple.
                Defaults to None.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            tuple[ReactionWheelStateEffectorStateDict, tuple[Tensor]]:
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
        angular_velocity = state_dict['dynamic_params']['angular_velocity']

        if self.max_torque.dim() != motor_torque.dim():
            motor_torque = motor_torque.unsqueeze(-2)

        torque_saturation_mask = self.max_torque > 0
        motor_torque = torch.where(
            torque_saturation_mask,
            torch.clamp(
                motor_torque,
                min=-self.max_torque,
                max=self.max_torque,
            ),
            motor_torque,
        )

        power_saturation_mask = ((self.max_power_efficiency > 0) & (torch.abs(
            motor_torque * angular_velocity) >= self.max_power_efficiency))
        motor_torque = torch.where(
            power_saturation_mask,
            torch.copysign(
                self.max_power_efficiency / angular_velocity,
                motor_torque,
            ),
            motor_torque,
        )

        speed_saturation_mask = (
            (self.max_angular_velocity > 0.) &
            (torch.abs(angular_velocity) >= self.max_angular_velocity) &
            (angular_velocity * motor_torque >= 0))
        motor_torque = torch.where(
            speed_saturation_mask,
            0.,
            motor_torque,
        )
        state_dict['current_torque'] = motor_torque

        battery_state_dict, state_dict = self.compute_power_usage(
            battery_state_dict=battery_state_dict, state_dict=state_dict)

        return state_dict, (battery_state_dict, )
