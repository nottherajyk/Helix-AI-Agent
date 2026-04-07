import numpy as np
from .base_agent import AbstractAgent

class RandomAgent(AbstractAgent):
    def act(self, observation: np.ndarray) -> np.ndarray:
        return self.action_space.sample()
