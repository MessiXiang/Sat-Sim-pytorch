import json

import torch
import numpy as np

from satsim.architecture import Timer
from satsim.data import Constellation, OrbitalElements
from satsim.simulation.task import TaskManager, Task, Tasks
from .satellite import RemoteSensingConstellation

from .constants import MAX_NUM_SATELLITES, MAX_NUM_TASKS

NUM_SATELLITES = 200
NUM_TASKS = 250
NUM_ENVS = 10


def split_envs(mode: str, sum_val: int, min_val: int,
               alpha: float) -> list[int]:
    """
    alpha: variability parameter, larger alpha means more balanced distribution
    """
    split_size = NUM_ENVS
    if mode == 'constellation':
        max_val = MAX_NUM_SATELLITES
    else:
        max_val = MAX_NUM_TASKS
    if sum_val < split_size * min_val or sum_val > split_size * max_val:
        raise ValueError("Constraints cannot be satisfied")

    max_attempts = 10000
    for _ in range(max_attempts):
        arr = []
        current_sum = 0
        for i in range(split_size - 1):
            val = np.random.randint(min_val, max_val + 1)
            arr.append(val)
            current_sum += val

        remaining = sum_val - current_sum
        if min_val <= remaining <= max_val:
            arr.append(remaining)
            return arr

    raise RuntimeError("Could not satisfy constraints within max attempts")


constellation = Constellation.sample(NUM_SATELLITES)
timer = Timer(1.)
constellation_dict = constellation.to_dict()

orbits = OrbitalElements.sample(NUM_SATELLITES)
orbits_dict = orbits.to_dicts()

tasks = Tasks.sample(NUM_TASKS)
tasks_dict = tasks.to_dict()

num_satellites_split = split_envs(mode='constellation',
                                  sum_val=NUM_SATELLITES,
                                  min_val=10,
                                  alpha=1.)

num_tasks_split = split_envs(mode='task',
                             sum_val=NUM_TASKS,
                             min_val=15,
                             alpha=1.)

saved_config = dict(constellation=constellation_dict,
                    orbits=orbits_dict,
                    tasks=tasks_dict,
                    num_satellites_split=num_satellites_split,
                    num_tasks_split=num_tasks_split)

constellation = RemoteSensingConstellation(
    timer=timer,
    constellation=constellation,
    orbits=orbits,
    use_battery=True,
)
torch.save(constellation.reset(), 'saved_state.pth')

with open('saved_config.json', 'w', encoding='utf-8') as f:
    json.dump(saved_config, f, indent=4, ensure_ascii=False)
