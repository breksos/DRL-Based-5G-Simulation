"""The decision process the DQN learns: build one TTI's allocation group by group.

Instead of choosing all ``n_rb`` assignments at once (an action space of
``n_ue ** n_rb``), the agent fills the TTI one resource-block group (RBG) at a
time. Each decision picks the UE for the next RBG, so the action that is
stored in the replay buffer is exactly the action that was executed.

The reward of a decision is the change in the scheduling objective it caused,
so the rewards of one TTI sum to ``R(final) − R(empty)``. ``R(empty)`` is a
constant, hence maximising the undiscounted return is the same as maximising
the per-TTI reward the environment reports.
"""

from __future__ import annotations

import numpy as np

from drl5g.channel import capacity_bps, drop_ues, snr_db
from drl5g.config import EnvConfig
from drl5g.env import objective_for
from drl5g.metrics import jain_index, starved_fraction

CAPACITY_REF_BPS = float(capacity_bps(30.0))
"""Per-RB rate at 30 dB; used to bring capacities to roughly unit scale."""

FEATURES_PER_UE = 4
"""Per UE: capacity, share of RBs so far, throughput vs fair share, served flag."""

GLOBAL_FEATURES = 4
"""Throughput term, Jain index, starved share, decisions left."""


class RBGAllocationTask:
    """Sequential allocation of one TTI.

    Observation layout (length ``4 · n_ue + 4``), grouped feature-major so a
    permutation-equivariant network can reshape it to ``(features, n_ue)``::

        [ capacity_k / C_ref                 for every UE
        | RBs given to k / n_rb              for every UE
        | throughput_k / fair share          for every UE  (fair share = T0 / n_ue)
        | 1 if k has been served, else 0     for every UE
        | min(T / T0, 1), Jain index, share of UEs not yet served,
          decisions left / decisions per TTI ]

    The global terms are the ingredients of the objective, so the network can
    tell how much one more RB group to a UE would change it.
    """

    def __init__(
        self,
        env_config: EnvConfig,
        rbg_size: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        if env_config.n_rb % rbg_size:
            raise ValueError("n_rb must be a multiple of rbg_size")
        self.env_config = env_config
        self.rbg_size = rbg_size
        self.n_ue = env_config.n_ue
        self.n_decisions = env_config.n_rb // rbg_size
        self.observation_size = FEATURES_PER_UE * self.n_ue + GLOBAL_FEATURES
        self._fair_share_bps = env_config.throughput_norm_mbps * 1e6 / self.n_ue
        self._rng = rng if rng is not None else np.random.default_rng()

        self._capacity = np.zeros(self.n_ue)
        self._rbs = np.zeros(self.n_ue)
        self._throughput = np.zeros(self.n_ue)
        self._choices: list[int] = []
        self._empty_objective = objective_for(env_config, np.zeros(self.n_ue))
        self._objective = self._empty_objective

    @property
    def objective(self) -> float:
        """Scheduling objective of the allocation built so far."""
        return self._objective

    @property
    def done(self) -> bool:
        return len(self._choices) == self.n_decisions

    @property
    def allocation(self) -> np.ndarray:
        """``(n_rb,)`` allocation built so far (complete once ``done``)."""
        return np.repeat(np.asarray(self._choices, dtype=np.int64), self.rbg_size)

    def reset(self, rb_capacity_bps: np.ndarray | None = None) -> np.ndarray:
        """Start a new TTI. Without capacities, drop a fresh set of UEs."""
        if rb_capacity_bps is None:
            distances, _ = drop_ues(self.n_ue, self.env_config.cell_radius, self._rng)
            rb_capacity_bps = capacity_bps(snr_db(distances))
        self._capacity = np.asarray(rb_capacity_bps, dtype=np.float64)
        self._rbs = np.zeros(self.n_ue)
        self._throughput = np.zeros(self.n_ue)
        self._choices = []
        self._objective = self._empty_objective
        return self.observe()

    def step(self, ue: int) -> tuple[np.ndarray, float, bool]:
        """Give the next RBG to ``ue``. Returns ``(observation, reward, done)``."""
        if self.done:
            raise RuntimeError("TTI already fully allocated; call reset()")
        self._choices.append(int(ue))
        self._rbs[ue] += self.rbg_size
        self._throughput[ue] += self.rbg_size * self._capacity[ue]

        objective = objective_for(self.env_config, self._throughput)
        reward = objective - self._objective
        self._objective = objective
        return self.observe(), reward, self.done

    def observe(self) -> np.ndarray:
        cfg = self.env_config
        total_mbps = self._throughput.sum() / 1e6
        decisions_left = (self.n_decisions - len(self._choices)) / self.n_decisions
        return np.concatenate(
            [
                self._capacity / CAPACITY_REF_BPS,
                self._rbs / cfg.n_rb,
                self._throughput / self._fair_share_bps,
                (self._throughput > 0).astype(np.float64),
                [
                    min(total_mbps / cfg.throughput_norm_mbps, 1.0),
                    jain_index(self._throughput),
                    starved_fraction(self._throughput),
                    decisions_left,
                ],
            ]
        ).astype(np.float32)
