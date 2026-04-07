from abc import ABC, abstractmethod
import numpy as np

class AbstractAgent(ABC):
    def __init__(self, observation_space, action_space, config=None):
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def act(self, observation: np.ndarray) -> np.ndarray:
        """Given observation, return action array of shape (4,)."""
        pass

    def learn(self, obs, action, reward, next_obs, terminated, info):
        """Optional: update internal policy. No-op by default."""
        pass

    def reset(self):
        """Called at episode start. No-op by default."""
        pass
