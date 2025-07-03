__all__ = [
    'ExternalForceTorque',
    'ExternalForceTorqueStateDict',
]

from typing import TypedDict

import torch
from torch import Tensor

from satsim.architecture import Module


class ExternalForceTorqueStateDict(TypedDict):

    external_force_body: Tensor  # External force in Body frame
    external_force_inertial: Tensor  # External force in Inertial frame
    external_torque_point_b_body: Tensor  # External torque about point B in Body frame


class ExternalForceTorque(Module[ExternalForceTorqueStateDict]):

    def reset(self) -> ExternalForceTorqueStateDict:

        state_dict = super().reset()
        zeros_vec = torch.zeros(3)

        state_dict.update({
            'external_force_body': zeros_vec.clone(),
            'external_force_inertial': zeros_vec.clone(),
            'external_torque_point_b_body': zeros_vec.clone(),
        })
        return state_dict

    def forward(
        self,
        state_dict: ExternalForceTorqueStateDict,
        *args,
        command_force_body_input: Tensor,
        command_force_inertial_input: Tensor,
        command_torque_body_input: Tensor,
        **kwargs,
    ) -> tuple[ExternalForceTorqueStateDict, tuple[Tensor, Tensor, Tensor]]:

        state_dict['external_force_body'] = state_dict[
            'external_force_body'] + command_force_body_input
        state_dict['external_force_inertial'] = state_dict[
            'external_force_inertial'] + command_force_inertial_input
        state_dict['external_torque_point_b_body'] = state_dict[
            'external_torque_point_b_body'] + command_torque_body_input

        outputs = (state_dict['external_force_body'],
                   state_dict['external_force_inertial'],
                   state_dict['external_torque_point_b_body'])

        return state_dict, outputs
