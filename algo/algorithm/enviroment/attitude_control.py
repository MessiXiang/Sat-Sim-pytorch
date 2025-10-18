from copy import deepcopy
from typing import NamedTuple, TypedDict

import einops
import torch

from algo.satellite import (AgentCommandTorqueConstellation,
                            RemoteSensingConstellation)
from satsim.architecture import Module, Timer, constants
from satsim.attitude_guidance.location_pointing import (
    LocationPointing, LocationPointingOutput, LocationPointingStateDict)
from satsim.data import (Constellation, ConstellationConfig, OrbitalElements,
                         OrbitDict, calculate_true_anomaly)
from satsim.simulation.base import BatteryStateDict
from satsim.simulation.gravity import Ephemeris
from satsim.simulation.power import SimpleBattery
from satsim.simulation.spacecraft import (SpacecraftStateDict,
                                          SpacecraftStateOutput)
from satsim.simulation.task import TaskDicts, Tasks
from satsim.utils import LLA2PCPF, move_to, mrp_to_rotation_matrix

DATA_DIM = 29


class AttitudeControlConstellationStateDict(TypedDict):
    _spacecraft: SpacecraftStateDict
    _battery: BatteryStateDict
    _location_pointing: LocationPointingStateDict


class AttitudeControlConstellation(
        RemoteSensingConstellation,
        Module[AttitudeControlConstellationStateDict]):

    def __init__(
        self,
        *args,
        constellation: Constellation,
        orbits: OrbitalElements,
        position_LP_P: torch.Tensor,
        **kwargs,
    ) -> None:
        self._n = constellation.num_satellite
        super(RemoteSensingConstellation, self).__init__(*args, **kwargs)
        super(AttitudeControlConstellation, self)._setup_spacecraft(
            constellation,
            'RK',
            orbits,
        )
        self.register_buffer('_position_LP_P', position_LP_P, persistent=False)
        self._battery = SimpleBattery(
            timer=self._timer,
            storage_capacity=torch.full([self._n], 20000),
            stored_charge_percentage_init=torch.full([self._n], 0.8),
        )
        self._location_pointing = LocationPointing(
            timer=self._timer,
            pointing_direction_B_B=torch.tensor(
                constellation.sensor.direction),
        )

    @property
    def position_LP_P(self) -> torch.Tensor:
        return self.get_buffer('_position_LP_P')

    def _calculate_location_inertial_position(
        self,
        earth_ephemeris: Ephemeris,
    ) -> torch.Tensor:
        direction_cosine_matrix_PN = earth_ephemeris[
            'direction_cosine_matrix_CN']
        position_PN_N = earth_ephemeris['position_CN_N']
        position_LP_N = torch.einsum(
            '...ij, ...i -> ...j',
            direction_cosine_matrix_PN,
            self._position_LP_P,
        )
        return position_PN_N + position_LP_N

    def forward(
        self,
        state_dict: AttitudeControlConstellationStateDict,
        *args,
        torque: torch.Tensor,
        **kwargs,
    ) -> tuple[AttitudeControlConstellationStateDict, tuple[
            torch.Tensor,
            LocationPointingOutput,
            Ephemeris,
    ]]:
        motor_torque = self._simple_motor_torque_assign(torque)
        battery_state_dict = state_dict['_battery']
        reaction_wheels_state_dict = state_dict['_spacecraft'][
            '_reaction_wheels']
        reaction_wheels_state_dict, (
            battery_state_dict, ) = self.reaction_wheels(
                state_dict=reaction_wheels_state_dict,
                battery_state_dict=battery_state_dict,
                motor_torque=motor_torque,
            )
        state_dict['_spacecraft'][
            '_reaction_wheels'] = reaction_wheels_state_dict

        spacecraft_state_dict = state_dict['_spacecraft']
        spacecraft_state_dict: SpacecraftStateDict
        spacecraft_state_output: SpacecraftStateOutput
        spacecraft_state_dict, spacecraft_state_output = self._spacecraft(
            spacecraft_state_dict)
        state_dict['_spacecraft'] = spacecraft_state_dict
        attitude_BN = spacecraft_state_dict['_hub']['dynamic_params'][
            'attitude_BN']
        angular_velocity_BN_B = spacecraft_state_dict['_hub'][
            'dynamic_params']['angular_velocity_BN_B']
        battery_state_dict = state_dict['_battery']

        sun_ephemeris: Ephemeris
        earth_ephemeris: Ephemeris
        _, (sun_ephemeris, ) = self.spice_interface(names=['SUN'])
        _, (earth_ephemeris, ) = self.spice_interface(names=['EARTH'])
        sun_ephemeris = move_to(
            sun_ephemeris,
            attitude_BN,
        )
        earth_ephemeris = move_to(
            earth_ephemeris,
            attitude_BN,
        )

        battery_state_dict, _ = self._battery(state_dict=battery_state_dict)
        state_dict['_battery'] = battery_state_dict

        position_LN_N = self._calculate_location_inertial_position(
            earth_ephemeris)

        pointing_guide_state_dict = state_dict['_location_pointing']
        pointing_guide_output: LocationPointingOutput
        pointing_guide_state_dict, pointing_guide_output = self._location_pointing(
            state_dict=pointing_guide_state_dict,
            position_LN_N=position_LN_N,
            position_BN_N=spacecraft_state_output.position_BN_N,
            attitude_BN=attitude_BN,
            angular_velocity_BN_B=angular_velocity_BN_B,
        )
        state_dict['_location_pointing'] = pointing_guide_state_dict
        attitude_BR: torch.Tensor = pointing_guide_output.attitude_BR
        return state_dict, (
            4 * torch.atan(attitude_BR.norm(dim=-1)),
            pointing_guide_output,
            earth_ephemeris,
        )


class CoordinateBuffer(NamedTuple):
    latitude: torch.Tensor
    longitude: torch.Tensor


class AttitudeControlEnviroment:

    def __init__(
        self,
        num_envs: int,
    ) -> None:
        self._num_envs = num_envs
        self._timer = Timer(1.)

    def _stochastic_init(self) -> None:
        self._constellations_data = Constellation.sample(self._num_envs)
        orbits = OrbitalElements.sample(self._num_envs)
        self._orbits_data = orbits
        self._tasks = Tasks.sample(self._num_envs)
        latitude, longitude = torch.tensor(
            [task.coordinate for task in self._tasks]).unbind(-1)
        self._latitude = latitude
        self._longitude = longitude
        position_LP_P = LLA2PCPF(
            latitude,
            longitude,
            torch.zeros_like(latitude),
            constants.REQ_EARTH,
            constants.REQ_EARTH,
        )
        self._simulator = AttitudeControlConstellation(
            timer=self._timer,
            constellation=self._constellations_data,
            orbits=orbits,
            position_LP_P=position_LP_P)

    def reset(
        self,
        init=False,
    ) -> None:
        if init:
            self._stochastic_init()
        self._timer.reset()
        self._simulator_state_dict = self._simulator.reset()

        return torch.cat(
            [self._get_observation(),
             torch.zeros(self._num_envs, 6)],
            dim=-1,
        )

    def _pick_dynamic_data(self) -> torch.Tensor:
        reaction_wheels_speed = self._simulator_state_dict['_spacecraft'][
            '_reaction_wheels']['dynamic_params']['angular_velocity'].clone(
            ).squeeze(-2)
        hub_dynam = self._simulator_state_dict['_spacecraft']['_hub'][
            'dynamic_params']
        angular_velocity = hub_dynam['angular_velocity_BN_B'].clone()
        attitude = hub_dynam['attitude_BN'].clone()
        if attitude.dim() == 1:
            attitude = attitude.expand_as(angular_velocity)

        battery_state_dict = self._simulator_state_dict['_battery']
        percentage = battery_state_dict['stored_charge_percentage'].clone(
        ).unsqueeze(-1)

        # b, 10
        return torch.cat(
            [
                reaction_wheels_speed,
                angular_velocity,
                attitude,
                percentage,
            ],
            dim=-1,
        )

    def _pick_static_data(self) -> torch.Tensor:
        reaction_wheel_data = self._constellations_data.reaction_wheels.normalized_static_data
        mass_property = self._constellations_data.normalized_mass_property

        # b, 13
        return torch.cat(
            [
                self._simulator.reaction_wheels.moment_of_inertia_wrt_spin.
                squeeze(-2),
                reaction_wheel_data,
                mass_property,
            ],
            dim=-1,
        )

    def _get_observation(self) -> torch.Tensor:
        observation = torch.cat(
            [self._pick_dynamic_data(),
             self._pick_static_data()],
            dim=-1,
        )
        return observation

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_dict: AttitudeControlConstellationStateDict
        attitude_info: LocationPointingOutput
        earth_ephemeris: Ephemeris
        self._simulator_state_dict, (
            angle_error,
            attitude_info,
            earth_ephemeris,
        ) = self._simulator(
            self._simulator_state_dict,
            torque=actions,
        )
        self._timer.step()
        observation = self._get_observation()
        observation = torch.cat(
            [
                observation, attitude_info.attitude_BR,
                attitude_info.angular_velocity_BR_B
            ],
            dim=-1,
        )

        battery_percentage = self._simulator_state_dict['_battery'][
            'stored_charge_percentage']
        battery_loss = 0.1 * (1 - battery_percentage).sum() / self._num_envs

        attitude_loss = torch.nn.functional.mse_loss(
            attitude_info.attitude_BR,
            torch.zeros_like(attitude_info.attitude_BR),
        ) + torch.nn.functional.mse_loss(
            attitude_info.angular_velocity_BR_B,
            torch.zeros_like(attitude_info.angular_velocity_BR_B),
        )

        loss = attitude_loss + battery_loss
        return observation, loss

    @torch.no_grad()
    def test_step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_dict: AttitudeControlConstellationStateDict
        attitude_info: LocationPointingOutput
        earth_ephemeris: Ephemeris
        self._simulator_state_dict, (
            angle_error,
            attitude_info,
            earth_ephemeris,
        ) = self._simulator(
            self._simulator_state_dict,
            torque=actions,
        )
        self._timer.step()
        observation = self._get_observation()
        observation = torch.cat(
            [
                observation, attitude_info.attitude_BR,
                attitude_info.angular_velocity_BR_B
            ],
            dim=-1,
        )

        battery_percentage = self._simulator_state_dict['_battery'][
            'stored_charge_percentage']

        return observation, angle_error, battery_percentage

    def state_dict(
        self
    ) -> tuple[
            list[OrbitDict],
            list[ConstellationConfig],
            AttitudeControlConstellationStateDict,
            TaskDicts,
    ]:
        return self._orbits_data.to_dicts(), self._constellations_data.to_dict(
        ), self._simulator_state_dict, self._tasks.to_dict()

    def load_state_dict(
        self,
        orbtis_data: list[OrbitDict],
        tasks_data: TaskDicts,
        constellation_data: list[ConstellationConfig],
        simulator_state_dict: AttitudeControlConstellationStateDict,
    ) -> None:
        orbits = OrbitalElements.from_dicts(orbtis_data)
        constellation = Constellation.from_dicts(constellation_data)
        self._constellations_data = constellation
        self._orbits_data = orbits
        self._tasks = Tasks.from_dict(tasks_data)
        latitude, longitude = torch.tensor(
            [task.coordinate for task in self._tasks]).unbind(-1)
        self._latitude = latitude
        self._longitude = longitude
        position_LP_P = LLA2PCPF(
            self._latitude,
            self._longitude,
            torch.zeros_like(self._latitude),
            constants.REQ_EARTH,
            constants.REQ_EARTH,
        )
        self._simulator = AttitudeControlConstellation(
            timer=self._timer,
            constellation=self._constellations_data,
            orbits=orbits,
            position_LP_P=position_LP_P)
        self._simulator_state_dict = simulator_state_dict

    def to(self, device: torch.device) -> None:
        self._simulator.to(device)
