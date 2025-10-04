__all__ = [
    'Task',
    'TaskManager',
]

import dataclasses
import random
from collections import UserList
from typing import NamedTuple, Self, TypedDict, cast

import torch

from satsim.architecture import Timer, constants


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
    finished: bool

    @classmethod
    def from_dict(cls, config: TaskDict) -> Self:
        return cls(
            id_=config['id'],
            release_time=config['release_time'],
            due_time=config['due_time'],
            duration=config['duration'],
            coordinate=config['coordinate'],
            finished=False,
        )

    def to_dict(self) -> TaskDict:
        d = dataclasses.asdict(self)
        d['id'] = d.pop('id_')
        return cast(TaskDict, d)

    def is_accessible(self, current_time: float) -> bool:
        return self.release_time <= current_time <= self.due_time and self.finished is False

    def finish(self) -> None:
        self.finished = True

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
            id_,
            release_time,
            due_time,
            duration,
            Coordinate(
                random.uniform(-90, 90),
                random.uniform(-180, 180),
            ),
            finished=False,
        )


class Tasks(UserList[Task]):

    def accessible_tasks(self, current_time) -> 'Tasks':
        return Tasks(
            [task for task in self if task.is_accessible(current_time)])

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

    def __init__(self, timer: Timer, tasks: list[Tasks]) -> None:
        self._timer = timer
        self._tasks = tasks
