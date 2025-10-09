import torch

from satsim.architecture import constants
from satsim.data import calculate_true_anomaly

from ..satellite import (RemoteSensingConstellation,
                         RemoteSensingConstellationStateDict)


def pick_and_normalize_dynamic_data(
    constellation: RemoteSensingConstellation,
    state_dict: RemoteSensingConstellationStateDict,
):
    reaction_wheels_speed = state_dict['_spacecraft']['_reaction_wheels'][
        'dynamic_params']['angular_velocity'].clone().detach().squeeze(-2)
    hub_dynam = state_dict['_spacecraft']['_hub']['dynamic_params']
    angular_velocity = hub_dynam['angular_velocity_BN_B'].clone().detach()
    attitude = hub_dynam['attitude_BN'].clone().detach()
    position_BP_N = hub_dynam['position_BP_N'].clone().detach()
    velocty_BP_N = hub_dynam['velocity_BP_N'].clone().detach()
    true_anomaly = calculate_true_anomaly(
        constants.MU_EARTH * 1e9,
        position_BP_N,
        velocty_BP_N,
    ).unsqueeze(-1)

    reaction_wheel_inertia = constellation.reaction_wheels.moment_of_inertia_wrt_spin
    reaction_wheel_inertia = torch.diagonal(
        reaction_wheel_inertia,
        dim1=1,
        dim2=2,
    )

    # data dim is 13
    return torch.cat(
        [
            reaction_wheels_speed,
            angular_velocity,
            attitude,
            true_anomaly,
            reaction_wheel_inertia,
        ],
        dim=-1,
    )
