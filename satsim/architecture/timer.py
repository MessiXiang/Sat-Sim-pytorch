class Timer:

    def __init__(
        self,
        dt: float = 0.01,
        start_time: float = 0.0,
    ) -> None:
        self._dt: float = dt
        self._start_time: float = start_time
        self._step_count: int = 0

    @property
    def step_count(self) -> int:
        return self.step_count

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def time(self) -> float:
        return self._start_time + self.step_count * self._dt

    def step(self) -> None:
        self.step_count += 1

    def reset(self) -> None:
        self.step_count = 0
