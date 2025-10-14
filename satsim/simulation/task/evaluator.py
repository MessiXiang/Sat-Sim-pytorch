__all__ = [
    'CompletionRateEvaluator',
    'WeightedCompletionRateEvaluator',
    'ProgressCompletionRateEvaluator',
    'WeighterProgressCompletionRateEvaluator',
    'PowerEvaluator',
    'TurnAroundTimeEvaluator',
    'Metrics',
    'EvaluatorMetrics',
]

from typing import TypedDict
from abc import ABC, abstractmethod
import torch

from ..task import TaskManager, Task, Tasks


class Metrics(TypedDict):
    final: torch.Tensor | None
    steps: list[torch.Tensor]


class EvaluatorMetrics(TypedDict):
    completion_rate: Metrics
    weighted_completion_rate: Metrics
    progress_completion_rate: Metrics
    progress_completion: Metrics
    weighted_progress_completion: Metrics
    power_consumption: Metrics
    turn_around_time: Metrics


class BaseEvaluator(ABC):

    def __init__(self, task_manager: TaskManager,
                 evaluator_metrics: EvaluatorMetrics):
        self.task_manager = task_manager
        self.evaluator_metrics = evaluator_metrics

    @abstractmethod
    @property
    def evaluate(self) -> torch.Tensor:
        pass

    @abstractmethod
    def on_step_end(self, **kwargs) -> None:
        pass

    @abstractmethod
    def on_run_end(self, **kwargs) -> None:
        pass


class CompletionRateEvaluator(BaseEvaluator):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def evaluate(self) -> torch.Tensor:
        return self.task_manager.num_finished_tasks / torch.tensor(
            self.task_manager.num_tasks)

    def on_step_end(self, **kwargs) -> None:
        self.evaluator_metrics['completion_rate']['steps'].append(
            self.evaluate)

    def on_run_end(self, **kwargs) -> None:
        self.evaluator_metrics['completion_rate']["final"] = self.evaluate


class WeightedCompletionRateEvaluator(BaseEvaluator):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def evaluate(self) -> torch.Tensor:
        all_time = self.task_manager.duration_all_tasks
        finished_time = torch.tensor([
            sum(task.duration for task in tasks)
            for tasks in self.task_manager.finished_tasks
        ])
        return finished_time / all_time

    def on_step_end(self, **kwargs) -> None:

        self.evaluator_metrics['weighted_completion_rate']['steps'].append(
            self.evaluate)

    def on_run_end(self, **kwargs) -> None:
        self.evaluator_metrics['weighted_completion_rate'][
            "final"] = self.evaluate


class ProgressCompletionRateEvaluator(BaseEvaluator):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def evaluate(self) -> torch.Tensor:
        complete_rate = self.task_manager.progress / self.task_manager.durations_flatten
        splits = torch.split(complete_rate, self.task_manager.num_tasks)
        average_rate = torch.tensor(
            [float(split.mean().item()) for split in splits])
        return average_rate

    def on_step_end(self, **kwargs) -> None:

        self.evaluator_metrics['progress_completion_rate']['steps'].append(
            self.evaluate)

    def on_run_end(self, **kwargs) -> None:
        self.evaluator_metrics['progress_completion_rate'][
            "final"] = self.evaluate


class WeighterProgressCompletionRateEvaluator(BaseEvaluator):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def evaluate(self) -> torch.Tensor:
        progress = self.task_manager.progress
        weights = self.task_manager.durations_flatten

        progress_splits = torch.split(progress, self.task_manager.num_tasks)
        weights_splits = torch.split(weights, self.task_manager.num_tasks)

        weighted_progress_per_env = torch.tensor([
            torch.sum(p * w).item()
            for p, w in zip(progress_splits, weights_splits)
        ])
        total_weight_per_env = torch.tensor(
            [torch.sum(w).item() for w in weights_splits])
        weighted_completion_rate = weighted_progress_per_env / total_weight_per_env
        return weighted_completion_rate

    def on_step_end(self, **kwargs) -> None:
        self.evaluator_metrics['weighted_progress_completion']['steps'].append(
            self.evaluate)

    def on_run_end(self, **kwargs) -> None:
        self.evaluator_metrics['weighted_progress_completion'][
            "final"] = self.evaluate


class PowerEvaluator(BaseEvaluator):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.power_used = 0.0

    @property
    def evaluate(self) -> torch.Tensor:
        raise NotImplementedError
        for sat_id, sat in self.environment.get_constellation().items():
            if dispatch_id[sat_id] != -1:
                self.power_used += sat.sensor.power

    # def on_step_end(self, dispatch_id: list, **kwargs) -> None:
    #     self.evaluator_metrics['power_consumption']['steps'].append(
    #         torch.tensor(self.power_used))

    def on_run_end(self, **kwargs) -> None:
        self.evaluator_metrics['power_consumption']['final'] = self.evaluate


class TurnAroundTimeEvaluator(BaseEvaluator):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.finish_time = torch.full((self.task_manager.num_total_tasks, ),
                                      float('inf'))
        self.minimum_finish_time = torch.tensor([
            task.release_time for task in self.task_manager.tasks_flatten
        ]) + self.task_manager.durations_flatten

    @property
    def evaluate(self) -> torch.Tensor:
        finished_tasks_mask = self.task_manager.finished_tasks_flags_flatten
        self.finish_time = torch.where(
            finished_tasks_mask,
            torch.minimum(self.finish_time,
                          torch.tensor(self.task_manager.timer.time)),
            self.finish_time)

        turn_around_time = torch.where(
            finished_tasks_mask, self.finish_time - self.minimum_finish_time,
            0.)
        splits = torch.split(turn_around_time, self.task_manager.num_tasks)
        turn_around_sums = torch.tensor(
            [float(split.sum().item()) for split in splits])

        zero_finished_tasks_mask = self.task_manager.num_finished_tasks == 0
        safe_num_finished = torch.where(zero_finished_tasks_mask, 1.,
                                        self.task_manager.num_finished_tasks)
        raw_average = turn_around_sums / safe_num_finished
        average_turn_around_time = torch.where(zero_finished_tasks_mask, 0.,
                                               raw_average)
        return average_turn_around_time

    def on_step_end(self, **kwargs) -> None:
        self.evaluator_metrics['turn_around_time']['steps'].append(
            self.evaluate)

    def on_run_end(self, **kwargs) -> None:
        self.evaluator_metrics['turn_around_time']["final"] = self.evaluate
