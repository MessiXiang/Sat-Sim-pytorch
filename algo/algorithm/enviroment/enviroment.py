__all__ = [
    'Enviroment',
]
from typing import Iterable, NamedTuple, TypedDict

import torch
import torch.nn.functional as F

from algo.satellite import (AgentCommandTorqueConstellation,
                            RemoteSensingConstellationStateDict)
from satsim.architecture import Timer, constants
from satsim.data import Constellation, OrbitalElements
from satsim.simulation.spacecraft import IntegrateMethod
from satsim.simulation.task import Tasks
from satsim.utils import LLA2PCPF

from ..utils import pick_dynamic_data
from .task_manager import TaskManager

MAX_TASKS_NUM = 312
MAX_SATELLITE_NUM = 64


def padding(v: torch.Tensor, split: Iterable[int],
            pad_len: int) -> torch.Tensor:
    split = tuple(split)
    batch_size = len(split)
    padding = torch.zeros(batch_size, pad_len, v.size(-1))
    split_v = torch.split(v, split)

    for i in range(batch_size):
        padding[i, :split[i], :] = split_v[i]

    return padding


def unpadding(v: torch.Tensor, split: Iterable[int]) -> torch.Tensor:
    split = tuple(split)
    pickup = []
    for i in range(v.size(0)):
        valid = v[i, :split[i], :]
        pickup.append(valid)
    return torch.cat(pickup, dim=0)


class Observation(NamedTuple):
    num_tasks: tuple[int, ...]
    num_satellites: tuple[int, ...]
    constellation_data: torch.Tensor  # b, ns, ds
    tasks_data: torch.Tensor  # b, nt, dt
    tasks_visibility: torch.Tensor  # b, nt


class ActionRestultBuffer(TypedDict):
    """Use to store action result for reward calculation
    """
    angle_error: torch.Tensor
    valid_task_progress_mask: torch.Tensor


TorqueAction = tuple[torch.Tensor, ...]


class Enviroment:

    def __init__(
        self,
        use_battery: bool,
        batch_size: int,
        integrate_method: IntegrateMethod = 'RK',
        simulation_step: int = 5,
    ) -> None:
        self._simulation_step = simulation_step
        self._use_battery = use_battery
        self._batch_size = batch_size
        self._integrate_method = integrate_method
        self._timer = Timer(1.)

    def _stochastic_init(self) -> None:
        num_satellite_per_env = torch.randint(
            5,
            MAX_SATELLITE_NUM,
            [self._batch_size],
        )
        num_tasks = torch.randint(
            10,
            MAX_TASKS_NUM,
            [self._batch_size],
        )
        num_satellite_per_env = tuple(num_satellite_per_env.tolist())
        total_satellite_num = sum(num_satellite_per_env)
        constellations = Constellation.sample(total_satellite_num)
        orbits = OrbitalElements.sample(total_satellite_num)
        tasks = tuple([Tasks.sample(n) for n in num_tasks.tolist()])

        self._constellations_data = constellations
        self._orbit_data = orbits

        self._task_manager = TaskManager(
            self._timer,
            tasks,
            num_satellite_per_env,
        )

        *_, latitude, longitude = self._task_manager.tasks.to_tensor().unbind(
            -1)
        position_LP_P = LLA2PCPF(
            latitude,
            longitude,
            torch.zeros_like(latitude),
            constants.REQ_EARTH * 1e3,
            constants.REQ_EARTH * 1e3,
        )

        self._simulator = AgentCommandTorqueConstellation(
            timer=self._timer,
            constellation=constellations,
            orbits=orbits,
            use_battery=self._use_battery,
            position_LP_P=position_LP_P,
            integrate_method=self._integrate_method,
        )

    def _get_observation(self) -> Observation:
        static_data = self._constellations_data.normalized_static_data
        dynamic_data = pick_dynamic_data(
            self._simulator,
            self._simulator_state_dict,
        )
        satellites_data = torch.cat(
            [static_data, dynamic_data],
            dim=-1,
        )
        satellites_data = padding(
            satellites_data,
            self._task_manager.num_satellites,
            pad_len=MAX_SATELLITE_NUM,
        )

        tasks_data = self._task_manager.get_task_progress_data()
        tasks_data = padding(tasks_data, self._task_manager.num_tasks,
                             MAX_TASKS_NUM)
        tasks_visibility = ~self._task_manager.has_completed & self._task_manager.tasks.is_accessible(
            self._timer.time)

        return Observation(
            self._task_manager.num_tasks,
            self._task_manager.num_satellites,
            satellites_data,
            tasks_data,
            tasks_visibility,
        )

    def _get_reward(self) -> torch.Tensor:
        # task_reward is a contact and step function which lead to a non-differentiable nature.
        task_reward = self._task_manager.progress.sum()

        battery_state_dict = self._simulator_state_dict['_power_supply'][
            '_battery']
        battery_reward = battery_state_dict['stored_charge_percentage']

        # use angle_error as a auxiliary reward for model to learn when task_reward is wherever zero
        angle_error = self._action_result_buffer['angle_error']
        valid_progress_mask = self._action_result_buffer[
            'valid_task_progress_mask']
        valid_angle_error = torch.where(valid_progress_mask, angle_error, 0.)
        valid_angle_error = torch.max(valid_angle_error, dim=0)[0]
        auxiliary_angle_error_reward = -valid_angle_error.sum() * 0.1

        return task_reward + battery_reward + auxiliary_angle_error_reward

    def _take_actions(self, actions: TorqueAction) -> None:
        """Use actions (torque per satellite) to update simulator state and task progress.
        Args:
            actions (TorqueAction): torque per satellite. size(b, max_satellite, 3)

        """
        for action in actions:

            action = unpadding(action, self._task_manager.num_satellites)

            self._simulator_state_dict, (
                is_visible,
                angle_error,
            ) = self._simulator(
                self._simulator_state_dict,
                cross_env_invisible_mask=self._task_manager.
                cross_env_invisible_mask,
                agent_commanded_torque=actions,
            )
            self._timer.step()

            valid_task_progress_mask = self._task_manager.step(is_visible)

        # use last state to calculate reward
        self._action_result_buffer = ActionRestultBuffer(
            angle_error=angle_error,
            valid_task_progress_mask=valid_task_progress_mask,
        )

    def reset(self) -> Observation:
        self._stochastic_init()

        self._simulator_state_dict = self._simulator.reset()
        self._timer.reset()
        return self._get_observation()

    def step(
        self,
        actions: TorqueAction,
    ) -> tuple[Observation, torch.Tensor, torch.Tensor]:

        self._take_actions(actions)
