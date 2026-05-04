"""Minimal CartPole-like env for DQN smoke tests. Internal use only."""
import numpy as np


class MockEnv:
    """4-dim state, 2 actions. Reward=1.0 per step. Done after 200 steps."""

    def __init__(self):
        self.state_dim = 4
        self.action_dim = 2
        self._step = 0

    def reset(self):
        self._step = 0
        return np.random.uniform(-0.05, 0.05, size=4).astype(np.float32)

    def step(self, action):
        self._step += 1
        next_state = np.random.uniform(-0.05, 0.05, size=4).astype(np.float32)
        done = self._step >= 200
        return next_state, 1.0, done, False, {}
