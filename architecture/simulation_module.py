import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class SimulationModule(nn.Module, ABC):

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @abstractmethod
    def simulation_step(self, dt: float) -> torch.Tensor:
        pass

    def forward(self, dt: float) -> torch.Tensor:
        return self.simulation_step(dt)

    def get_simulation_state(self) -> Dict[str, Any]:
        return self.state_dict()

    def load_simulation_state(self, state: Dict[str, Any]):
        self.load_state_dict(state)

    @abstractmethod
    def reset_simulation_state(self):
        pass
