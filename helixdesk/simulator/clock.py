import numpy as np

class SimClock:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.minutes: float = 0.0
        self.tick_minutes: float = 0.0

    def tick(self) -> float:
        """Advance the clock between 8 and 15 simulated minutes"""
        delta = float(self.rng.integers(8, 16))
        self.minutes += delta
        self.tick_minutes = delta
        return self.minutes

    def reset(self, seed: int | None = None):
        """Reset the sim clock back to 0"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.minutes = 0.0
        self.tick_minutes = 0.0
