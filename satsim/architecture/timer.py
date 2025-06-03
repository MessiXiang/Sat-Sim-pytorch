from typing import TypedDict


# TODO: add timer state dict
class TimerStateDict(TypedDict):
    step_count: int


class Timer:

    def __init__(
        self,
        dt: float = 0.01,
        start_time: float = 0.0,
    ) -> None:
        self.dt = dt
        self._start_time = start_time

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def dt(self) -> float:
        return self._dt

    @dt.setter
    def dt(self, value: float) -> None:
        assert value > 0, "dt must be a float superior than 0"
        self._dt = value

    @property
    def time(self) -> float:
        return self._start_time + self._step_count * self._dt

    def step(self) -> None:
        self._step_count += 1

    def reset(self) -> None:
        self._step_count = 0

    def state_dict(self) -> TimerStateDict:
        return dict(step_count=self._step_count)

    def load_state_dict(self, state_dict: TimerStateDict) -> None:
        self._step_count = state_dict["step_count"]
