from typing import Dict, Any, Union, Callable
import math

import torch
import torch.nn as nn

from satsim.architecture import (
    SimulationModule,
    BSKError,
    Message
)

class InputFeeder(SimulationModule):
    def __init__(self, name:str , value: Message = None):
        super().__init__(name)
        self.value = value

    def simulation_step(self, timestamp: float, message: Message) -> Message:

        value = self.value 
        self.value = None
        return value

    def set_value(self, value: Dict[str, torch.Tensor]):
        
        self.value = value
    
    def reset_simulation_state(self, timestamp):
        pass

class OutputLogger(SimulationModule):
    def __init__(self, name):
        super().__init__(name)

    def simulation_step(self, timestamp, message):
        if timestamp == 0:
            self.recorder = message
            self.recorder = {
                key: value.unsqueeze(0)
                for key, value in self.recorder.items()
            }
        else:
            assert all([key in self.recorder for key in message])
            
            self.recorder = {
                key: torch.cat([self.recorder[key], message[key].unsqueeze(0)], dim=0) 
                for key in self.recorder
            }
        
    def reset_simulation_state(self, timestamp):
        pass

    def get_recorder(self):
        return self.recorder
    
        
