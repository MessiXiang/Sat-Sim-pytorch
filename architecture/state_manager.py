from typing import Dict, Any, Optional
import copy

class StateManager:

    def __init__(self, auto_save: bool = True):
        self.checkpoints: Dict[int, Dict[str, Any]] = {}
        self.auto_save = auto_save

    def save_state(self, state_dict: Dict[str, Any], step: int):
        self.checkpoints[step] = copy.deepcopy(state_dict)

    def load_state(self, step: int) -> Optional[Dict[str, Any]]:
        if step in self.checkpoints:
            return copy.deepcopy(self.checkpoints[step])
        return None

    def clear(self):
        self.checkpoints.clear()
