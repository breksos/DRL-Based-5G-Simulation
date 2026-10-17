"""Double DQN with a permutation-equivariant Q-network."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from drl5g.config import DQNConfig
from drl5g.rl.allocation import FEATURES_PER_UE, GLOBAL_FEATURES


class EquivariantQNetwork(nn.Module):
    """Scores every UE with the same small MLP.

    Each UE's input is its own features, the mean and max of every UE's
    features (context), and the global features. Because the weights are
    shared, relabelling the UEs relabels the Q-values in the same way, and one
    network handles any number of UEs.

    Input: ``(batch, FEATURES_PER_UE · n_ue + GLOBAL_FEATURES)`` in the layout
    produced by :class:`~drl5g.rl.allocation.RBGAllocationTask`.
    Output: ``(batch, n_ue)``.
    """

    def __init__(self, hidden_sizes: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = 3 * FEATURES_PER_UE + GLOBAL_FEATURES
        for size in hidden_sizes:
            layers += [nn.Linear(width, size), nn.ReLU()]
            width = size
        layers.append(nn.Linear(width, 1))
        self.scorer = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch, length = obs.shape
        n_ue = (length - GLOBAL_FEATURES) // FEATURES_PER_UE
        per_ue = obs[:, :-GLOBAL_FEATURES].reshape(batch, FEATURES_PER_UE, n_ue).transpose(1, 2)
        mean = per_ue.mean(dim=1, keepdim=True).expand_as(per_ue)
        peak = per_ue.amax(dim=1, keepdim=True).expand_as(per_ue)
        shared = obs[:, -GLOBAL_FEATURES:].unsqueeze(1).expand(batch, n_ue, GLOBAL_FEATURES)
        return self.scorer(torch.cat([per_ue, mean, peak, shared], dim=-1)).squeeze(-1)


class ReplayBuffer:
    """Fixed-size ring buffer of transitions stored in pre-allocated arrays."""

    def __init__(self, capacity: int, obs_size: int, rng: np.random.Generator) -> None:
        self._rng = rng
        self._obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._cursor = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        i = self._cursor
        self._obs[i], self._actions[i], self._rewards[i] = obs, action, reward
        self._next_obs[i], self._dones[i] = next_obs, float(done)
        self._cursor = (i + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        idx = self._rng.integers(0, self._size, size=batch_size)
        return tuple(
            torch.from_numpy(a[idx])
            for a in (self._obs, self._actions, self._rewards, self._next_obs, self._dones)
        )


@dataclass
class EpsilonSchedule:
    """Linear decay from ``start`` to ``end`` over ``decay_steps``, then constant."""

    start: float
    end: float
    decay_steps: int

    def __call__(self, step: int) -> float:
        fraction = min(step / max(self.decay_steps, 1), 1.0)
        return self.start + fraction * (self.end - self.start)


class DQNAgent:
    """Double DQN: online network picks the next action, target network scores it.

    Uses Huber loss, gradient-norm clipping and hard target updates.
    """

    def __init__(
        self,
        obs_size: int,
        n_actions: int,
        config: DQNConfig,
        total_steps: int,
        seed: int = 0,
    ) -> None:
        self.config = config
        self.n_actions = n_actions
        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.q_net = EquivariantQNetwork(config.hidden_sizes)
        self.target_net = EquivariantQNetwork(config.hidden_sizes)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(config.buffer_size, obs_size, self._rng)
        self.epsilon_schedule = EpsilonSchedule(
            config.epsilon_start,
            config.epsilon_end,
            int(config.epsilon_decay_fraction * total_steps),
        )
        self.steps = 0

    @property
    def epsilon(self) -> float:
        return self.epsilon_schedule(self.steps)

    def act(self, obs: np.ndarray, explore: bool = True) -> int:
        if explore and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(obs)))

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        return self.q_net(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()

    def observe(self, obs, action: int, reward: float, next_obs, done: bool) -> float | None:
        """Store a transition, take one gradient step if due, return the loss."""
        self.buffer.add(obs, action, reward, next_obs, done)
        self.steps += 1
        if len(self.buffer) < max(self.config.learning_starts, self.config.batch_size):
            return None
        loss = self._update()
        if self.steps % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return loss

    def _update(self) -> float:
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.config.batch_size)
        q_taken = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.q_net(next_obs).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_obs).gather(1, next_actions).squeeze(1)
            target = rewards + self.config.gamma * (1.0 - dones) * next_q

        loss = nn.functional.smooth_l1_loss(q_taken, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        return float(loss.item())
