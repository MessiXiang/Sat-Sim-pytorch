class SimulationClock:

    def __init__(self, dt: float = 0.01, start_time: float = 0.0):
        self.dt = dt
        self.start_time = start_time
        self.step_count = 0

    def step(self):
        self.step_count += 1

    def reset(self):
        self.step_count = 0

    @property
    def time(self) -> float:
        return self.start_time + self.step_count * self.dt

    @property
    def steps(self) -> int:
        return self.step_count
