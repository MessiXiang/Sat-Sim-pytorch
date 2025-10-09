import json

import torch
from matplotlib import pyplot as plt

from satsim.architecture import Timer, constants
from satsim.data import Constellation, OrbitalElements
from satsim.utils import LLA2PCPF

from .satellite import (RemoteSensingConstellation,
                        RemoteSensingConstellationStateDict)

with open('saved_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

constellation = Constellation.from_dicts(config['constellation'])
timer = Timer(1.)
orbits = OrbitalElements.from_dicts(config['orbits'])
tasks = config['tasks']
latitude = torch.tensor(tasks['latitude'])
longitude = torch.tensor(tasks['longitude'])

constellation = RemoteSensingConstellation(timer=timer,
                                           constellation=constellation,
                                           orbits=orbits,
                                           use_battery=True)
position_LP_P = LLA2PCPF(
    latitude,
    longitude,
    torch.zeros_like(latitude),
    constants.REQ_EARTH * 1e3,
    constants.REQ_EARTH * 1e3,
)
state_dict = torch.load('saved_state.pth')
constellation.setup_target(position_LP_P)

timer.reset()

angle_errors = []
stored_charges = []
while timer.time < 180:
    state_dict: RemoteSensingConstellationStateDict
    state_dict, (
        is_filming,
        spacecraft_state_output,
        stored_charge,
    ) = constellation(
        state_dict,
        charging=torch.ones(50, dtype=torch.bool),
        sensor_turn_on=torch.zeros(50, dtype=torch.bool),
    )
    timer.step()
    stored_charges.append(stored_charge)

    attitude_BR = state_dict['_sun_guide']['_location_poiniting'][
        'attitude_BR_old']
    angle_error = 4 * torch.atan(attitude_BR.norm(dim=-1))

    angle_errors.append(angle_error)

angle_errors_list = torch.stack(angle_errors, dim=-1).tolist()
stored_charges_list = torch.stack(stored_charges, dim=-1).tolist()
plt.clf()
for angle_error in angle_errors_list:
    plt.plot(angle_error)
plt.xlabel('Timestep')
plt.ylabel('Angle Error (rad)')
plt.title('Angle Error over Time')
plt.savefig('angle_error_test.png')

plt.clf()
for stored_charge in stored_charges_list:
    plt.plot(stored_charge)
plt.xlabel('Timestep')
plt.ylabel('Battery Percentage (%)')
plt.title('Battery Percentage over Time')
plt.savefig('battery_percentage_test.png')
