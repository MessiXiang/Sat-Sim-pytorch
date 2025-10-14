__all__ = [
    'Task',
    'Tasks',
    'TaskManager',
]

import dataclasses
import random
import math
from collections import UserList
from typing import NamedTuple, Self, TypedDict, cast

import torch

from satsim.architecture import Timer, constants

# 0--unreleased--)+[--accessible--+--finished-)+[-finished --->
#              release         complete        due
#
# 0--unreleased--)+[---------accessible--------)+[--failed----->
#              release                         due
#
# closed = succeed + failed


class Coordinate(NamedTuple):
    latitude: float
    longitude: float


class TaskDict(TypedDict):
    id: int
    release_time: float
    due_time: float
    duration: float
    coordinate: Coordinate


TaskDicts = list[TaskDict]


@dataclasses.dataclass
class Task:
    id_: int
    release_time: float
    due_time: float
    duration: float
    coordinate: Coordinate

    @classmethod
    def from_dict(cls, config: TaskDict) -> Self:
        return cls(id_=config['id'],
                   release_time=config['release_time'],
                   due_time=config['due_time'],
                   duration=config['duration'],
                   coordinate=config['coordinate'])

    def to_dict(self) -> TaskDict:
        d = dataclasses.asdict(self)
        d['id'] = d.pop('id_')
        return cast(TaskDict, d)

    @property
    def data(self) -> list[float]:
        return [
            self.release_time,
            self.due_time,
            self.duration,
            *self.coordinate,
        ]

    @classmethod
    def sample(cls, id_: int) -> Self:
        # e.g. 20
        duration = random.randint(15, 60)
        # e.g. 3540 \in [0, 3540]
        release_time = random.randint(0,
                                      constants.MAX_TIME_STEP - duration * 3)
        # e.g. 3600 \in [3600, 3600]
        due_time = random.randint(release_time + duration * 3,
                                  constants.MAX_TIME_STEP)
        return cls(
            id_, release_time, due_time, duration,
            Coordinate(
                random.uniform(-math.pi / 2, math.pi / 2),
                random.uniform(-math.pi, math.pi),
            ))


class Tasks(UserList[Task]):

    def unreleased_flags(self, current_time) -> torch.Tensor:
        return torch.tensor(
            [current_time < task.release_time for task in self])

    def accessible_flags(self, current_time,
                         finished_flags: torch.Tensor) -> torch.Tensor:
        return torch.tensor([
            task.release_time <= current_time < task.due_time and not finished
            for task, finished in zip(self, finished_flags)
        ])

    def failed_flags(self, current_time,
                     finished_flags: torch.Tensor) -> torch.Tensor:
        return torch.tensor([
            task.due_time <= current_time and not finished
            for task, finished in zip(self, finished_flags)
        ])

    def closed_flags(self, current_time) -> torch.Tensor:
        return torch.tensor([task.due_time <= current_time for task in self])

    def _filter_tasks(self, flags: torch.Tensor) -> 'Tasks':
        return Tasks(task for task, flag in zip(self, flags) if flag)

    def to_tensor(self) -> torch.Tensor:
        data = torch.tensor([task.data for task in self])
        return data

    @property
    def coordinate(self) -> torch.Tensor:
        return torch.tensor([task.coordinate for task in self])

    @property
    def num_tasks(self) -> int:
        return len(self)

    @classmethod
    def sample(cls, n: int) -> 'Tasks':
        return cls([Task.sample(i) for i in range(n)])

    @classmethod
    def from_dict(cls, tasks: TaskDicts) -> 'Tasks':
        return cls([Task.from_dict(task) for task in tasks])

    def to_dict(self) -> TaskDicts:
        return [task.to_dict() for task in self]


class TaskManager:

    def __init__(self, timer: Timer, tasks: list[Tasks],
                 num_satellites: list[int]) -> None:
        """
        Manages tasks for multiple environments.
        :param tasks: list of Tasks for each environment.
        :param num_satellites: list of number of satellites for each environment.
        """
        self.timer = timer
        self.tasks = tasks
        self.num_satellites = num_satellites

        self.tasks_flatten = [task for tasks in self.tasks for task in tasks]
        self.durations_flatten = torch.tensor(
            [task.duration for task in self.tasks_flatten])
        self.num_tasks = [len(task) for task in tasks]
        self.num_total_tasks = sum(self.num_tasks)

        num_tasks_offset = [0]
        for i in range(1, len(self.num_tasks)):
            num_tasks_offset.append(num_tasks_offset[-1] +
                                    self.num_tasks[i - 1])
        self.num_tasks_offset = num_tasks_offset

        self.progress = torch.zeros(
            self.num_total_tasks,
            dtype=torch.uint32,
        )
        self._finished_tasks_flags_flatten = torch.zeros(
            self.num_total_tasks,
            dtype=torch.bool,
        )

    @property
    def unreleased_tasks(self) -> list[Tasks]:
        return [
            tasks._filter_tasks(tasks.unreleased_flags(self.timer.time))
            for tasks in self.tasks
        ]

    @property
    def unreleased_tasks_flags(self) -> list[torch.Tensor]:
        return [
            tasks.unreleased_flags(self.timer.time) for tasks in self.tasks
        ]

    @property
    def accessible_tasks(self) -> list[Tasks]:
        splits = torch.split(self._finished_tasks_flags_flatten,
                             self.num_tasks)
        return [
            tasks._filter_tasks(
                tasks.accessible_flags(self.timer.time, splits[idx]))
            for idx, tasks in enumerate(self.tasks)
        ]

    @property
    def accessible_tasks_flags(self) -> list[torch.Tensor]:
        splits = torch.split(self._finished_tasks_flags_flatten,
                             self.num_tasks)
        return [
            tasks.accessible_flags(self.timer.time, splits[idx])
            for idx, tasks in enumerate(self.tasks)
        ]

    @property
    def finished_tasks(self) -> list[Tasks]:
        splits = torch.split(self._finished_tasks_flags_flatten,
                             self.num_tasks)
        return [
            tasks._filter_tasks(splits[idx])
            for idx, tasks in enumerate(self.tasks)
        ]

    @property
    def finished_tasks_flags_flatten(self) -> torch.Tensor:
        return self._finished_tasks_flags_flatten

    @property
    def failed_tasks(self) -> list[Tasks]:
        splits = torch.split(self._finished_tasks_flags_flatten,
                             self.num_tasks)
        return [
            tasks._filter_tasks(
                tasks.failed_flags(self.timer.time, splits[idx]))
            for idx, tasks in enumerate(self.tasks)
        ]

    @property
    def failed_tasks_flags(self) -> list[torch.Tensor]:
        splits = torch.split(self._finished_tasks_flags_flatten,
                             self.num_tasks)
        return [
            tasks.failed_flags(self.timer.time, splits[idx])
            for idx, tasks in enumerate(self.tasks)
        ]

    @property
    def closed_tasks(self) -> list[Tasks]:
        return [
            tasks._filter_tasks(tasks.closed_flags(self.timer.time))
            for tasks in self.tasks
        ]

    @property
    def closed_tasks_flags(self) -> list[torch.Tensor]:
        return [tasks.closed_flags(self.timer.time) for tasks in self.tasks]

    @property
    def all_tasks(self) -> list[Tasks]:
        return self.tasks

    @property
    def all_closed(self) -> torch.Tensor:
        return torch.tensor([
            len(closed) == num_tasks
            for closed, num_tasks in zip(self.closed_tasks, self.num_tasks)
        ])

    @property
    def is_idle(self) -> torch.Tensor:
        return torch.tensor(
            [len(accessibles) == 0 for accessibles in self.accessible_tasks])

    @property
    def num_accessible_tasks(self) -> torch.Tensor:
        return torch.tensor(
            [len(accessible) for accessible in self.accessible_tasks])

    @property
    def num_finished_tasks(self) -> torch.Tensor:
        splits = torch.split(self._finished_tasks_flags_flatten,
                             self.num_tasks)
        return torch.tensor([int(split.sum().item()) for split in splits])

    @property
    def num_closed_tasks(self) -> torch.Tensor:
        return torch.tensor([len(closed) for closed in self.closed_tasks])

    @property
    def duration_all_tasks(self) -> torch.Tensor:
        splits = torch.split(self.durations_flatten, self.num_tasks)
        return torch.tensor([int(split.sum().item()) for split in splits])

    def step(self, is_filming: torch.Tensor, actions: torch.Tensor) -> None:
        """
        :param is_filming: (num_total_tasks, num_total_satellites) for all tasks and all satellites in all environments.
            True if the point is filming by the satellite.
            NOTE: Each satellite must be assigned to a pointting target, position_LP_P=0 for standby.
        :param actions: (num_total_satellites,)  -2 for standby, -1 for charging, >=0 for filming task_id, max=max_num_tasks.
        """
        env_indices = torch.cat([
            torch.full((num_sats, ), idx, device=actions.device)
            for idx, num_sats in enumerate(self.num_satellites)
        ])

        task_ids = actions.clone()
        filming_mask = task_ids >= 0
        filming_task_ids = task_ids[filming_mask]
        env_offsets = torch.tensor(
            self.num_tasks_offset,
            device=actions.device)[env_indices[filming_mask]]
        filming_task_ids += env_offsets  # (num_filming_tasks,) in global task ids

        filming_sat_indices = torch.nonzero(filming_mask).squeeze(dim=1)

        if filming_task_ids.numel() == 0:
            return

        is_visible = torch.zeros_like(self._finished_tasks_flags_flatten,
                                      dtype=torch.bool)
        task_visibility = is_filming[filming_task_ids, filming_sat_indices]
        is_visible.scatter_(dim=0,
                            index=filming_task_ids,
                            src=task_visibility,
                            reduce='or')

        flatten_accessible_flag = torch.cat(self.accessible_tasks_flags)
        is_visible = is_visible & flatten_accessible_flag

        self.progress = self.progress + is_visible.int()
        finished_mask = self.progress >= self.durations_flatten
        self._finished_tasks_flags_flatten |= finished_mask

    def padding(self, flags: list[torch.Tensor],
                max_num_tasks: int) -> torch.Tensor:
        """
        flags: (num_total_tasks,) True for real tasks, False for padding

        return: (num_envs, max_num_tasks)
        """
        padded_flags = []
        for flag in flags:
            padding_flag = torch.zeros(max_num_tasks - flag.shape[0],
                                       dtype=torch.bool,
                                       device=flag.device)
            flag = torch.cat([flag, padding_flag], dim=0)
            padded_flags.append(flag)
        return torch.stack(padded_flags)
