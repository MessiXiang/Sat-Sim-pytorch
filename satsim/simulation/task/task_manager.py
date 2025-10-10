__all__ = [
    'Task',
    'TaskManager',
]

import dataclasses
import random
import math
from collections import UserList
from typing import NamedTuple, Self, TypedDict, cast

from numpy import pad
import torch

from satsim.architecture import Timer, constants

# 0--unreleased--)+[--accessible--+--succeeded-)+[-succeeded --->
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
                         succeeded_flags: torch.Tensor) -> torch.Tensor:
        return torch.tensor([
            task.release_time <= current_time < task.due_time and not finished
            for task, finished in zip(self, succeeded_flags)
        ])

    def failed_flags(self, current_time,
                     succeeded_flags: torch.Tensor) -> torch.Tensor:
        return torch.tensor([
            task.due_time <= current_time and not finished
            for task, finished in zip(self, succeeded_flags)
        ])

    def closed_flags(self, current_time) -> torch.Tensor:
        return torch.tensor([task.due_time <= current_time for task in self])

    def _filter_tasks(self, flags: torch.Tensor) -> 'Tasks':
        return Tasks(task for task, flag in zip(self, flags) if flag)

    def to_tensor(self) -> torch.Tensor:
        data = torch.tensor([task.data for task in self])
        return data

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
                 nums_spacecrafts: int) -> None:
        """
        Manages tasks for multiple environments.
        """
        self._timer = timer
        self._tasks = tasks
        self._num_spacecrafts = nums_spacecrafts

        self._flatten_tasks = [task for tasks in self._tasks for task in tasks]
        self._durations = torch.tensor(
            [task.duration for task in self._flatten_tasks])
        self._num_tasks = [len(task) for task in tasks]
        self._num_total_tasks = sum(self._num_tasks)

        self._progress = torch.zeros(
            self._num_total_tasks,
            dtype=torch.uint32,
        )
        self._succeeded_tasks_flags = torch.zeros(
            self._num_total_tasks,
            dtype=torch.bool,
        )

    @property
    def unreleased_tasks(self) -> list[Tasks]:
        return [
            tasks._filter_tasks(tasks.unreleased_flags(self._timer.time))
            for tasks in self._tasks
        ]

    @property
    def unreleased_tasks_flags(self) -> list[torch.Tensor]:
        return [
            tasks.unreleased_flags(self._timer.time) for tasks in self._tasks
        ]

    @property
    def accessible_tasks(self) -> list[Tasks]:
        splits = torch.split(self._succeeded_tasks_flags, self._num_tasks)
        return [
            tasks._filter_tasks(
                tasks.accessible_flags(self._timer.time, splits[idx]))
            for idx, tasks in enumerate(self._tasks)
        ]

    @property
    def accessible_tasks_flags(self) -> list[torch.Tensor]:
        splits = torch.split(self._succeeded_tasks_flags, self._num_tasks)
        return [
            tasks.accessible_flags(self._timer.time, splits[idx])
            for idx, tasks in enumerate(self._tasks)
        ]

    @property
    def succeeded_tasks(self) -> list[Tasks]:
        splits = torch.split(self._succeeded_tasks_flags, self._num_tasks)
        return [
            tasks._filter_tasks(splits[idx])
            for idx, tasks in enumerate(self._tasks)
        ]

    @property
    def succeeded_tasks_flags(self) -> torch.Tensor:
        return self._succeeded_tasks_flags

    @property
    def failed_tasks(self) -> list[Tasks]:
        splits = torch.split(self._succeeded_tasks_flags, self._num_tasks)
        return [
            tasks._filter_tasks(
                tasks.failed_flags(self._timer.time, splits[idx]))
            for idx, tasks in enumerate(self._tasks)
        ]

    @property
    def failed_tasks_flags(self) -> list[torch.Tensor]:
        splits = torch.split(self._succeeded_tasks_flags, self._num_tasks)
        return [
            tasks.failed_flags(self._timer.time, splits[idx])
            for idx, tasks in enumerate(self._tasks)
        ]

    @property
    def closed_tasks(self) -> list[Tasks]:
        return [
            tasks._filter_tasks(tasks.closed_flags(self._timer.time))
            for tasks in self._tasks
        ]

    @property
    def closed_tasks_flags(self) -> list[torch.Tensor]:
        return [tasks.closed_flags(self._timer.time) for tasks in self._tasks]

    @property
    def num_total_tasks(self) -> int:
        return self._num_total_tasks

    @property
    def all_tasks(self) -> list[Tasks]:
        return self._tasks

    @property
    def progress(self) -> torch.Tensor:
        return self._progress

    @property
    def all_closed(self) -> torch.Tensor:
        return torch.tensor([
            len(closed) == num_tasks
            for closed, num_tasks in zip(self.closed_tasks, self._num_tasks)
        ])

    @property
    def is_idle(self) -> torch.Tensor:
        return torch.tensor(
            [len(accessibles) == 0 for accessibles in self.accessible_tasks])

    @property
    def num_all_tasks(self) -> int:
        return self._num_total_tasks

    @property
    def num_accessible_tasks(self) -> list[int]:
        return [len(accessible) for accessible in self.accessible_tasks]

    @property
    def num_succeeded_tasks(self) -> list[int]:
        splits = torch.split(self._succeeded_tasks_flags, self._num_tasks)
        return [int(split.sum().item()) for split in splits]

    def step(self, is_filming: torch.Tensor, actions: torch.Tensor) -> None:
        """
        is_filming: (num_total_tasks, num_total_spacecrafts) for all tasks and all spacecrafts in all environments.
            True if the point is filming by the spacecraft.
            NOTE: Every spacecraft must be assigned to a pointting target, position_LP_P=0 for do nothing.
        actions: (num_total_spacecrafts,)  -2 for do nothing, -1 for charging, >=0 for filming task_id.        
        """
        filming_mask = actions >= 0
        filming_task_ids = actions[filming_mask]
        filming_sat_indices = torch.nonzero(filming_mask).squeeze(dim=1)

        if filming_task_ids.numel() == 0:
            return

        is_visible = torch.zeros_like(self._succeeded_tasks_flags,
                                      dtype=torch.bool)
        task_visibility = is_filming[filming_task_ids, filming_sat_indices]
        is_visible.scatter_(dim=0, index=filming_task_ids, src=task_visibility, reduce='or')

        flatten_accessible_flag = torch.cat(self.accessible_tasks_flags)
        is_visible = is_visible & flatten_accessible_flag

        self._progress = self._progress + is_visible.int()
        succeeded_mask = self._progress >= self._durations
        self._succeeded_tasks_flags |= succeeded_mask

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
