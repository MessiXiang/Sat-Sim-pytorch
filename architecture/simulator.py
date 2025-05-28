import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from .simulation_clock import SimulationClock
from .state_manager import StateManager
from .simulation_module import SimulationModule

class Simulator:

    def __init__(self, module: SimulationModule, dt: float = 0.01, auto_save: bool = True):
        self.clock = SimulationClock(dt)
        self.state_manager = StateManager(auto_save)
        self.module: SimulationModule = module

    def step(self, save_state: bool = None) -> torch.Tensor:
        result = self.module(self.clock.dt)
        self.clock.step()

        should_save = save_state if save_state is not None else self.state_manager.auto_save
        if should_save:
            self.state_manager.save_state(self.module.get_simulation_state(), self.clock.steps)
            print(f"Step {self.clock.steps}: result = {result}")
        return result

    def run(self, steps: int, save_interval: Optional[int] = None):
        for i in range(steps):
            should_save = (save_interval is not None and
                          self.clock.steps % save_interval == 0)
            self.step(save_state=should_save)

    def reset(self):
        self.clock.reset()
        self.state_manager.clear()
        self.module.reset_simulation_state()

    def save_checkpoint(self):
        self.state_manager.save_state(self.module.get_simulation_state(), self.clock.steps)

    def load_checkpoint(self, step: int):
        state = self.state_manager.load_state(step)
        if state is None:
            raise ValueError(f"No checkpoint found for step {step}")
        self.clock.step_count = step
        self.module.load_simulation_state(state)

    @property
    def time(self) -> float:
        return self.clock.time

    @property
    def steps(self) -> int:
        return self.clock.steps

    @property
    def dt(self) -> float:
        return self.clock.dt
