import json

import torch
import numpy as np
from matplotlib import pyplot as plt

from .constants import MAX_TIME_STEPS, MAX_NUM_SATELLITES, MAX_NUM_TASKS, EVALUATION_INTERVAL
from satsim.architecture import Timer, constants
from satsim.data import Constellation, OrbitalElements
from satsim.utils import LLA2PCPF
from satsim.simulation.task import TaskManager, Task, Tasks
from satsim.simulation.task import (CompletionRateEvaluator,
                                    WeightedCompletionRateEvaluator,
                                    WeighterProgressCompletionRateEvaluator,
                                    ProgressCompletionRateEvaluator,
                                    PowerEvaluator, TurnAroundTimeEvaluator,
                                    EvaluatorMetrics, Metrics)
from .satellite import (RemoteSensingConstellation,
                        RemoteSensingConstellationStateDict)


def actions_to_commands(
        actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """  
    :param actions: (num_statellites, ) int, -2 for standby, -1 for charging, >=0 for filming task_id, max=max_num_tasks
    :return: 
        charging: (num_statellites, ), bool
        target_position: (num_statellites, 3), float 

    """
    charging_mask = actions == -1
    filming_mask = actions >= 0

    charging = torch.zeros_like(actions, dtype=torch.bool)
    charging[charging_mask] = True

    target_position = torch.zeros((num_satellites, 3), dtype=torch.float32)
    actions[filming_mask] += torch.tensor(task_manager.num_tasks_offset)
    target_position[filming_mask] = position_LP_P[actions[filming_mask]]

    return (charging, target_position)


with open('saved_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

constellation = Constellation.from_dicts(config['constellation'])
num_satellites = constellation.num_satellite
num_satellites_split = config['num_satellites_split']

timer = Timer(1.)
orbits = OrbitalElements.from_dicts(config['orbits'])

constellation = RemoteSensingConstellation(timer=timer,
                                           constellation=constellation,
                                           orbits=orbits,
                                           use_battery=True)

tasks: Tasks
tasks = Tasks.from_dict(config['tasks'])
num_tasks = tasks.num_tasks
num_tasks_split = config['num_tasks_split']

tasks_split = []
cumsum = [0] + list(np.cumsum(num_tasks_split))
for i in range(len(num_tasks_split)):
    start = cumsum[i]
    end = cumsum[i + 1]
    tasks_split.append(Tasks(tasks[start:end]))

coordinate = tasks.coordinate
latitude, longitude = coordinate[:, 0], coordinate[:, 1]
position_LP_P = LLA2PCPF(
    latitude,
    longitude,
    torch.zeros_like(latitude),
    constants.REQ_EARTH * 1e3,
    constants.REQ_EARTH * 1e3,
)

state_dict = torch.load('saved_state.pth')

task_manager = TaskManager(timer=timer,
                           tasks=tasks_split,
                           num_satellites=num_satellites_split)

evaluators = [
    CompletionRateEvaluator(task_manager=task_manager),
    ProgressCompletionRateEvaluator(task_manager=task_manager),
    WeighterProgressCompletionRateEvaluator(task_manager=task_manager),
    WeightedCompletionRateEvaluator(task_manager=task_manager),
    TurnAroundTimeEvaluator(task_manager=task_manager),
    PowerEvaluator(task_manager=task_manager),
]

evaluator_metrics = EvaluatorMetrics(
    completion_rate=Metrics(final=None, steps=[]),
    weighted_completion_rate=Metrics(final=None, steps=[]),
    progress_completion_rate=Metrics(final=None, steps=[]),
    progress_completion=Metrics(final=None, steps=[]),
    weighted_progress_completion=Metrics(final=None, steps=[]),
    power_consumption=Metrics(final=None, steps=[]),
    turn_around_time=Metrics(final=None, steps=[]),
)

timer.reset()

while timer.time < MAX_TIME_STEPS:
    actions = model.predict(state_dict)  # and tasks conditions
    charging, target_position = actions_to_commands(actions)
    constellation.setup_target(target_position)

    state_dict: RemoteSensingConstellationStateDict
    state_dict, (
        is_filming,
        spacecraft_state_output,
        stored_charge,
    ) = constellation(
        state_dict,
        charging=charging,
        sensor_turn_on=torch.ones(50, dtype=torch.bool),
    )

    task_manager.step(is_filming, actions)

    if int(timer.time) % EVALUATION_INTERVAL == 0:
        for evaluator in evaluators:
            evaluator.on_step_end()

    timer.step()

for evaluator in evaluators:
    evaluator.on_run_end()
