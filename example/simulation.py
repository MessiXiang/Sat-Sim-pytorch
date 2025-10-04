import argparse
import json
from email.mime import base
from re import L
from typing import Any

import matplotlib.pyplot as plt
import torch
import tqdm

from satsim.architecture import Timer, constants
from satsim.data.orbits import OrbitDict
from satsim.enviroment.ground_location import AccessState
from satsim.utils.lla import LLA2PCPF

from .simulator import (BatteryConfig, ConstellationConfig,
                        GroundMappingConfig, MRPControlConfig,
                        ReactionWheelConfig, RemoteSensingConstellation,
                        SolarPanelConfig)


def make_constellation_config(config: dict[str, Any]) -> ConstellationConfig:
    orbits: list[OrbitDict] = config['orbits']
    satellites = config['satellites']
    orbits = [{
        'true_anomaly': sat['true_anomaly'],
        **orbit
    } for sat, orbit in zip(satellites, orbits)]

    reaction_wheel = ReactionWheelConfig(
        mech_to_elec_efficiency=[
            sat['reaction_wheels'][0]['efficiency'] for sat in satellites
        ],
        base_power=[sat['reaction_wheels'][0]['power'] for sat in satellites],
        init_speed=[[rw['rw_speed_init'] for rw in sat['reaction_wheels']]
                    for sat in satellites],
    )

    solar_panel = SolarPanelConfig(
        panel_normal_B_B=[
            sat['solar_panel']['direction'] for sat in satellites
        ],
        panel_area=[sat['solar_panel']['area'] for sat in satellites],
        panel_efficiency=[
            sat['solar_panel']['efficiency'] for sat in satellites
        ],
    )

    ground_mapping = GroundMappingConfig(half_field_of_view=[
        sat['sensor']['half_field_of_view'] for sat in satellites
    ], )

    mrp_control = MRPControlConfig(
        k=[sat['mrp_control']['k'] for sat in satellites],
        ki=[sat['mrp_control']['ki'] for sat in satellites],
        p=[sat['mrp_control']['p'] for sat in satellites],
        integral_limit=[
            sat['mrp_control']['integral_limit'] for sat in satellites
        ],
    )
    battery = BatteryConfig(
        capacity=[sat['battery']['capacity'] for sat in satellites],
        percentage=[sat['battery']['percentage'] for sat in satellites],
    )

    config = ConstellationConfig(
        mass=[sat['mass'] for sat in satellites],
        inertia=[tuple(sat['inertia']) for sat in satellites],
        reaction_wheel=reaction_wheel,
        orbits=orbits,
        ground_mapping=ground_mapping,
        solar_panel=solar_panel,
        mrp_control=mrp_control,
        battery=battery)

    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--constellation', type=str)
    parser.add_argument('--tasksets', type=str)
    parser.add_argument('--state-dict', type=str, default=None)
    parser.add_argument('--use-battery', action='store_true', default=False)
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        default=False,
    )
    parser.add_argument('--grad', action='store_true', default=False)
    parser.add_argument('--time', type=int, default=180)
    args = parser.parse_args()

    if args.use_gpu:
        torch.set_default_device('cuda:0')

    with open(args.constellation, 'r') as f:
        config = json.load(f)

    config = make_constellation_config(config)

    timer = Timer(1.)
    simulator = RemoteSensingConstellation(timer=timer,
                                           config=config,
                                           use_battery=args.use_battery)
    if args.state_dict is not None:
        simulator_state_dict = torch.load(args.state_dict)
    else:
        simulator_state_dict = simulator.reset()

    timer.reset()

    with open(args.tasksets, 'r') as f:
        tasksets = json.load(f)
    lla = tasksets[0]['coordinate']
    latitude = torch.tensor([lla[0]])
    longitude = torch.tensor([lla[1]])
    position_LN_N = LLA2PCPF(
        latitude,
        longitude,
        torch.tensor([0.]),
        constants.REQ_EARTH * 1e3,
        constants.REQ_EARTH * 1e3,
    )
    if args.grad:
        grad_record = torch.ones(1, requires_grad=True)
        position_LN_N = grad_record * position_LN_N

    simulator.setup_target(
        simulator_state_dict,
        position_LN_N,
    )

    p_bar = tqdm.tqdm(total=args.time / timer.dt)
    access_state: AccessState
    angle_error: torch.Tensor
    battery_percentage: torch.Tensor
    angle_errors = []
    battery = []
    while timer.time < args.time:
        simulator_state_dict, (
            angle_error,
            mapping_access_state,
            battery_percentage,
        ) = simulator(state_dict=simulator_state_dict)
        timer.step()
        p_bar.update(1)
        angle_errors.append(angle_error.cpu())
        battery.append(battery_percentage.cpu())
    p_bar.close()

    angle_errors_list = torch.stack(angle_errors, dim=-1).tolist()
    battery = torch.stack(battery, dim=-1).tolist()

    if args.grad:
        torch.stack(angle_errors, dim=-1).sum().backward()
        print(grad_record.grad)

    # print(min(angle_errors))
    for angle_error in angle_errors_list:

        plt.plot(angle_error)
    plt.xlabel('Timestep')
    plt.ylabel('Angle Error (rad)')
    plt.title('Angle Error over Time')

    plt.savefig('angle_error.png')

    # for b in battery:
    #     plt.plot(b)
    # plt.xlabel('Timestep')
    # plt.ylabel('Battery Percentage %')
    # plt.title('Percentage overtime')
    # p_bar.close()
    # plt.savefig('battery.png')
