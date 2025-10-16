__all__ = [
    'TaskManager',
]
from typing import Iterable

import torch

from satsim.architecture import Timer
from satsim.simulation.task import Tasks


class TaskManager:

    def __init__(
        self,
        timer: Timer,
        tasks: Iterable[Tasks],
        num_satellites: Iterable[int],
    ) -> None:
        """
        Manages tasks for multiple environments.
        Assumes all environments have the same number of tasks.
        """
        self._timer = timer
        self._tasks = tuple(tasks)

        self._flatten_tasks = Tasks(
            [task for tasks in self._tasks for task in tasks])

        self._progress = torch.zeros(self.num_total_tasks, )
        self._num_satellites = tuple(num_satellites)

    @property
    def tasks_per_env(self) -> tuple[Tasks, ...]:
        return self._tasks

    @property
    def tasks(self) -> Tasks:
        return self._flatten_tasks

    @property
    def num_tasks(self) -> tuple[int, ...]:
        return tuple(len(tasks) for tasks in self._tasks)

    @property
    def num_satellites(self) -> tuple[int, ...]:
        return self._num_satellites

    @property
    def num_total_tasks(self) -> int:
        return len(self._flatten_tasks)

    @property
    def num_total_satellites(self) -> int:
        return sum(self._num_satellites)

    @property
    def cross_env_invisible_mask(self) -> torch.Tensor:  # nt, ns
        mask = torch.zeros(
            self.num_total_tasks,
            self.num_total_satellites,
            dtype=torch.bool,
        )
        task_id_start = 0
        satellite_id_start = 0
        for num_task, num_satellite in zip(self.num_tasks,
                                           self._num_satellites):
            env_task_mask = torch.zeros(self.num_total_tasks, dtype=torch.bool)
            env_task_mask[task_id_start:task_id_start + num_task] = True

            env_satellite_mask = torch.zeros(
                self.num_total_satellites,
                dtype=torch.bool,
            )
            env_satellite_mask[satellite_id_start:satellite_id_start +
                               num_satellite] = True

            env_visible_mask = torch.einsum(
                'i,j-> ij',
                env_task_mask,
                env_satellite_mask,
            )
            mask = mask | env_visible_mask

            task_id_start += num_task
            satellite_id_start += num_satellite

        return mask

    @property
    def progress(self) -> torch.Tensor:
        return self._progress

    @property
    def has_completed(self) -> torch.Tensor:
        return torch.tensor([
            self.progress[i] >= task.duration
            for i, task in enumerate(self.tasks)
        ])

    def step(self, is_visible: torch.Tensor) -> torch.Tensor:
        """
        is_visible: (num_total_tasks, num_total_spacecrafts) for all tasks and all spacecrafts in all environments.
                    True if the task is visible by assigned spacecraft.
        """
        is_visible = is_visible & self.cross_env_invisible_mask
        is_visible = is_visible.any(1)  # (num_total_tasks,)
        accessible_mask = self.tasks.is_accessible(self._timer.time)
        success_mask = self.has_completed

        valid_task_progress = is_visible & accessible_mask.unsqueeze(
            -1) & ~success_mask.unsqueeze(dim=-1)

        self._progress = self._progress + valid_task_progress.int(
        ) * self._timer.dt
        duration = torch.tensor([task.duration for task in self.tasks])
        self._progress = torch.clamp(
            self._progress,
            max=duration,
        )

        return valid_task_progress

    def get_task_progress_data(self) -> torch.Tensor:
        _, due_time, duration, *coordinate = self.tasks.to_tensor().unbind(
            dim=-1)
        time_left = due_time - self._timer.time
        progress_left = torch.clamp(duration - self.progress, min=0.)
        task_progress_data = torch.stack(
            [*coordinate, time_left, progress_left, due_time, duration],
            dim=-1)
        return task_progress_data
