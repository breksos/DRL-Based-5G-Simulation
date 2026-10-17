"""Classical reference schedulers."""

from __future__ import annotations

import numpy as np

from drl5g.config import EnvConfig
from drl5g.env import objective_for
from drl5g.schedulers.base import LastInputCache


class RandomScheduler:
    """Each RB goes to a uniformly random UE. The "no intelligence" floor."""

    name = "Random"

    def __init__(self, n_ue: int, n_rb: int, seed: int | None = None) -> None:
        self.n_ue, self.n_rb = n_ue, n_rb
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def allocate(self, rb_capacity_bps: np.ndarray) -> np.ndarray:
        return self._rng.integers(0, self.n_ue, size=self.n_rb)

    def reset(self) -> None:
        pass


class RoundRobinScheduler:
    """Deals RBs out in turn, starting one UE later every TTI.

    Every UE gets ``n_rb / n_ue`` RBs per TTI (±1), regardless of its channel.
    """

    name = "RoundRobin"

    def __init__(self, n_ue: int, n_rb: int) -> None:
        self.n_ue, self.n_rb = n_ue, n_rb
        self._offset = 0

    def allocate(self, rb_capacity_bps: np.ndarray) -> np.ndarray:
        allocation = (self._offset + np.arange(self.n_rb)) % self.n_ue
        self._offset = (self._offset + self.n_rb) % self.n_ue
        return allocation

    def reset(self) -> None:
        self._offset = 0


class MaxRateScheduler:
    """Max C/I: every RB goes to the UE with the best channel. The throughput ceiling."""

    name = "MaxRate"

    def __init__(self, n_ue: int, n_rb: int) -> None:
        self.n_ue, self.n_rb = n_ue, n_rb

    def allocate(self, rb_capacity_bps: np.ndarray) -> np.ndarray:
        return np.full(self.n_rb, int(np.argmax(rb_capacity_bps)))

    def reset(self) -> None:
        pass


class ProportionalFairScheduler:
    """Textbook proportional fair with an exponentially averaged served rate.

    Each RB goes to ``argmax_k r_k / T_k``, where ``r_k`` is the per-RB rate
    and ``T_k`` the UE's average served throughput. ``T_k`` is updated for every
    UE at the end of each TTI (unscheduled UEs average in a zero), and the
    within-TTI estimate includes the RBs already handed out, so one TTI is
    spread over several UEs. Averages start at the equal-share rate, which
    avoids a cold-start burst towards whichever UE happens to be first.

    With the static channel used here, PF converges to an equal number of RBs
    per UE: there is no fading for it to exploit.
    """

    name = "PropFair"

    def __init__(self, n_ue: int, n_rb: int, averaging_window: float = 100.0) -> None:
        self.n_ue, self.n_rb = n_ue, n_rb
        self._ema = 1.0 / averaging_window
        self._avg_rate: np.ndarray | None = None

    def allocate(self, rb_capacity_bps: np.ndarray) -> np.ndarray:
        rate = np.asarray(rb_capacity_bps, dtype=np.float64)
        if self._avg_rate is None:
            self._avg_rate = rate * self.n_rb / self.n_ue

        served = np.zeros(self.n_ue)
        allocation = np.empty(self.n_rb, dtype=np.int64)
        for rb in range(self.n_rb):
            projected_avg = (1.0 - self._ema) * self._avg_rate + self._ema * served
            ue = int(np.argmax(rate / np.maximum(projected_avg, 1e-9)))
            allocation[rb] = ue
            served[ue] += rate[ue]

        self._avg_rate = (1.0 - self._ema) * self._avg_rate + self._ema * served
        return allocation

    def reset(self) -> None:
        self._avg_rate = None


class GreedyObjectiveScheduler:
    """Hands out RBs in groups, each to the UE that raises the objective most.

    It knows the reward function in closed form, which no learning agent does,
    so it serves as a model-based reference point for the DQN rather than as a
    deployable baseline.
    """

    name = "Greedy"

    def __init__(self, env_config: EnvConfig, rbg_size: int = 1) -> None:
        if env_config.n_rb % rbg_size:
            raise ValueError("n_rb must be a multiple of rbg_size")
        self.n_ue, self.n_rb, self.rbg_size = env_config.n_ue, env_config.n_rb, rbg_size
        self._env_config = env_config
        self._cache = LastInputCache()

    def allocate(self, rb_capacity_bps: np.ndarray) -> np.ndarray:
        return self._cache.get_or_compute(rb_capacity_bps, self._greedy_fill)

    def _greedy_fill(self, rb_capacity_bps: np.ndarray) -> np.ndarray:
        group_rate = self.rbg_size * np.asarray(rb_capacity_bps, dtype=np.float64)
        throughput = np.zeros(self.n_ue)
        allocation = []
        for _ in range(self.n_rb // self.rbg_size):
            candidates = throughput + np.diag(group_rate)  # row k: next group goes to UE k
            ue = int(np.argmax([objective_for(self._env_config, c) for c in candidates]))
            throughput[ue] += group_rate[ue]
            allocation.append(ue)
        return np.repeat(allocation, self.rbg_size)

    def reset(self) -> None:
        pass
