"""
Baseline RB schedulers.
Interface (matches team contract): scheduler.allocate(state) -> action
  state  : np.array shape (2*N_UE + N_RB,)  [snrs | throughputs | rb_occ]
  action : np.array shape (N_RB,), dtype int, each element in 0..N_UE-1
"""

import numpy as np
from env.channel_model import capacity_bps


class _BaseScheduler:
    def __init__(self, n_ue: int, n_rb: int):
        self.n_ue = n_ue
        self.n_rb = n_rb

    def allocate(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self):
        pass


class RandomScheduler(_BaseScheduler):
    """Assigns each RB to a uniformly random UE. Lower-bound reference."""

    def __init__(self, n_ue: int, n_rb: int, seed: int = None):
        super().__init__(n_ue, n_rb)
        self._rng = np.random.default_rng(seed)

    def allocate(self, state: np.ndarray) -> np.ndarray:
        return self._rng.integers(0, self.n_ue, size=self.n_rb)


class RoundRobinScheduler(_BaseScheduler):
    """Cycles through UEs in order: RB_0→UE_0, RB_1→UE_1, … (mod N_UE)."""

    def __init__(self, n_ue: int, n_rb: int):
        super().__init__(n_ue, n_rb)
        self._counter = 0

    def allocate(self, state: np.ndarray) -> np.ndarray:
        action = np.array([(self._counter + i) % self.n_ue for i in range(self.n_rb)], dtype=int)
        self._counter = (self._counter + 1) % self.n_ue
        return action

    def reset(self):
        self._counter = 0


class ProportionalFairScheduler(_BaseScheduler):
    """
    Per-RB Proportional Fair: each RB independently picks
    argmax(instant_rate / avg_rate); the chosen UE's avg_rate is
    EMA-updated before the next RB is allocated within the same step.
    Lazy-initialised on first allocate call to avoid scale mismatch (bps vs. 1.0).
    """

    def __init__(self, n_ue: int, n_rb: int, alpha: float = 0.001):
        super().__init__(n_ue, n_rb)
        self._alpha = alpha
        self._avg_rates = None  # lazy-init on first allocate

    def allocate(self, state: np.ndarray) -> np.ndarray:
        snrs = state[:self.n_ue].astype(np.float64)
        instant = capacity_bps(snrs)                      # (N_UE,) bps
        if self._avg_rates is None:
            self._avg_rates = instant.mean() * np.ones(self.n_ue, dtype=np.float64)

        action = np.empty(self.n_rb, dtype=int)
        for rb in range(self.n_rb):
            priority = instant / (self._avg_rates + 1e-9)
            chosen = int(np.argmax(priority))
            action[rb] = chosen
            self._avg_rates[chosen] = (
                (1.0 - self._alpha) * self._avg_rates[chosen]
                + self._alpha * instant[chosen]
            )
        return action

    def reset(self):
        self._avg_rates = None
