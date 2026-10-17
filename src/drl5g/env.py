"""Gymnasium environment for single-cell downlink resource-block scheduling."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from drl5g.channel import capacity_bps, drop_ues, snr_db
from drl5g.config import EnvConfig
from drl5g.metrics import jain_index, per_ue_throughput, scheduling_objective

SNR_OBS_RANGE_DB = (-20.0, 40.0)


def objective_for(config: EnvConfig, throughputs_bps: np.ndarray) -> float:
    """The scheduling objective with the weights of ``config``."""
    return scheduling_objective(
        throughputs_bps,
        config.alpha,
        config.beta,
        config.throughput_norm_mbps,
        config.starvation_penalty,
    )


class FiveGEnv(gym.Env):
    """One gNB, ``n_ue`` static users and ``n_rb`` resource blocks.

    Each ``step`` is one TTI: the action assigns every RB to a UE and the reward
    is :func:`drl5g.metrics.scheduling_objective` of the resulting throughputs.
    UEs are re-dropped on every ``reset``; within an episode the channel is fixed.

    Observation (``float32``, length ``2 · n_ue``)::

        [ SNR_1 … SNR_N (dB) | throughput_1 … throughput_N of the last TTI (Mbit/s) ]

    Action: ``MultiDiscrete([n_ue] * n_rb)``; entry ``i`` is the UE served on RB ``i``.
    """

    metadata = {"render_modes": []}  # noqa: RUF012 - gymnasium convention

    def __init__(self, config: EnvConfig, seed: int | None = None) -> None:
        super().__init__()
        self.config = config
        n_ue, n_rb = config.n_ue, config.n_rb

        snr_low, snr_high = SNR_OBS_RANGE_DB
        max_tp_mbps = n_rb * float(capacity_bps(snr_high)) / 1e6
        self.observation_space = spaces.Box(
            low=np.array([snr_low] * n_ue + [0.0] * n_ue, dtype=np.float32),
            high=np.array([snr_high] * n_ue + [max_tp_mbps] * n_ue, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([n_ue] * n_rb)

        self._rng = np.random.default_rng(seed)
        self._distances = np.zeros(n_ue)
        self._angles = np.zeros(n_ue)
        self._snr_db = np.zeros(n_ue)
        self._rb_capacity = np.zeros(n_ue)
        self._last_throughput = np.zeros(n_ue)
        self._step_count = 0

    # -- public read-only state, used by schedulers and the GUI -----------------
    @property
    def n_ue(self) -> int:
        return self.config.n_ue

    @property
    def n_rb(self) -> int:
        return self.config.n_rb

    @property
    def distances(self) -> np.ndarray:
        return self._distances

    @property
    def angles(self) -> np.ndarray:
        return self._angles

    @property
    def snr_db(self) -> np.ndarray:
        return self._snr_db

    @property
    def rb_capacity_bps(self) -> np.ndarray:
        """Rate (bit/s) each UE would get from one RB in the current TTI."""
        return self._rb_capacity

    @property
    def last_throughput_bps(self) -> np.ndarray:
        return self._last_throughput

    @property
    def step_count(self) -> int:
        return self._step_count

    # -- gymnasium API ----------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._distances, self._angles = drop_ues(self.n_ue, self.config.cell_radius, self._rng)
        self._snr_db = snr_db(self._distances)
        self._rb_capacity = capacity_bps(self._snr_db)
        self._last_throughput = np.zeros(self.n_ue)
        self._step_count = 0
        return self._observation(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        throughput = per_ue_throughput(action, self._rb_capacity)
        self._last_throughput = throughput
        self._step_count += 1

        cfg = self.config
        reward = objective_for(cfg, throughput)
        info = {
            "throughput_bps": throughput,
            "throughput_total_bps": float(throughput.sum()),
            "fairness": jain_index(throughput),
        }
        truncated = self._step_count >= cfg.max_steps
        return self._observation(), reward, False, truncated, info

    def _observation(self) -> np.ndarray:
        obs = np.concatenate([self._snr_db, self._last_throughput / 1e6]).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)
