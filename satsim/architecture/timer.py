__all__ = [
    'Timer',
]

from typing import TypedDict

import todd


class StateDict(TypedDict):
    step: int


class Timer(todd.utils.StateDictMixin):

    def __init__(
        self,
        dt: float = 0.01,
        start_time: float = 0.0,
    ) -> None:
        assert dt > 0, "dt must be greater than 0"
        self._dt = dt
        self._start_time = start_time

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def step_count(self) -> int:
        return self._step

    def reset(self) -> None:
        self._step = 0

    @property
    def time(self) -> float:
        return self._start_time + self._step * self._dt

    def step(self) -> None:
        self._step += 1

    def state_dict(self, *args, **kwargs) -> StateDict:
        return StateDict(step=self._step)

    def load_state_dict(
        self,
        state_dict: StateDict,
        *args,
        **kwargs,
    ) -> todd.utils.Keys | None:
        self._step = state_dict['step']
