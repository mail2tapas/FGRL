
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FogSchedulingEnv(gym.Env):
    def __init__(self, dag):
        super().__init__()
        self.tasks=list(dag.nodes)
        self.current=0
        self.action_space=spaces.Discrete(4)
        self.observation_space=spaces.Box(low=0, high=100, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.current=0
        return self._state(), {}

    def _state(self):
        remaining=len(self.tasks)-self.current
        return np.array([
            remaining,
            self.current,
            remaining/max(len(self.tasks),1),
            0.5,
            0.7,
            0.3
        ], dtype=np.float32)

    def step(self, action):
        critical_bonus = 1.0 if self.current < max(1,len(self.tasks)//5) else 0.2
        energy_penalty = action * 0.05
        reward = critical_bonus - energy_penalty
        self.current += 1
        done = self.current >= len(self.tasks)
        return self._state(), reward, done, False, {}
