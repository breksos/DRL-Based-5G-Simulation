"""
5G downlink Gymnasium environment.
State  : [snr_1..snr_N, throughput_1..throughput_N, rb_occupancy_1..rb_N_RB]
Action : np.array of shape (N_RB,), each element in {0 .. N_UE-1}
Reward : alpha * sum(throughput) + beta * Jain's Fairness Index
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.channel_model import snr_db, capacity_bps, random_ue_distances, RB_BANDWIDTH_HZ


class FiveGEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_ue: int = 10, n_rb: int = 50,
                 cell_radius: float = 500.0,
                 alpha: float = 0.5, beta: float = 0.5,
                 max_steps: int = 200,
                 seed: int | None = None):
        super().__init__()
        self.n_ue = n_ue
        self.n_rb = n_rb
        self.cell_radius = cell_radius
        self.alpha = alpha
        self.beta = beta
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)

        obs_dim = 2 * n_ue + n_rb
        # SNR [0,30], throughput_norm [0, ~100 Mbps], rb_occupancy [0,1]
        high = np.array([30.0] * n_ue + [200.0] * n_ue + [1.0] * n_rb, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0.0, high=high, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([n_ue] * n_rb)

        self.mobility = False          # enable ±10 m random walk between episodes

        self._distances: np.ndarray = np.zeros(n_ue, dtype=np.float32)
        self._snrs: np.ndarray = np.zeros(n_ue, dtype=np.float32)
        self._throughputs: np.ndarray = np.zeros(n_ue, dtype=np.float32)
        self._rb_occupancy: np.ndarray = np.zeros(n_rb, dtype=np.float32)
        self._step_count: int = 0

    @property
    def distances(self) -> np.ndarray:
        return self._distances

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.mobility and self._distances.any():
            walk = self._rng.uniform(-10.0, 10.0, size=self.n_ue)
            distances = np.clip(self._distances + walk, 50.0, self.cell_radius)
        else:
            distances = random_ue_distances(
                self.n_ue, d_min=50.0, d_max=self.cell_radius, rng=self._rng
            )
        self._distances = distances.astype(np.float32)
        self._snrs = snr_db(distances).astype(np.float32)
        self._throughputs = np.zeros(self.n_ue, dtype=np.float32)
        self._rb_occupancy = np.zeros(self.n_rb, dtype=np.float32)
        self._step_count = 0

        return self._build_state(), {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=int)

        # Per-UE throughput: sum of capacities of assigned RBs
        ue_throughput = np.zeros(self.n_ue, dtype=np.float32)
        rb_occupancy = np.zeros(self.n_rb, dtype=np.float32)

        for rb_idx, ue_idx in enumerate(action):
            ue_idx = int(ue_idx) % self.n_ue
            cap = float(capacity_bps(self._snrs[ue_idx], RB_BANDWIDTH_HZ))
            ue_throughput[ue_idx] += cap
            rb_occupancy[rb_idx] = ue_idx / max(self.n_ue - 1, 1)  # normalised

        self._throughputs = ue_throughput
        self._rb_occupancy = rb_occupancy

        reward = float(self._compute_reward(ue_throughput))

        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.max_steps

        info = {
            "throughput_total": float(ue_throughput.sum()),
            "fairness": float(self._jains_index(ue_throughput)),
        }
        return self._build_state(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _build_state(self) -> np.ndarray:
        raw = np.concatenate([
            self._snrs,
            self._throughputs / 1e6,   # normalise to Mbps
            self._rb_occupancy,
        ]).astype(np.float32)
        return np.clip(raw, self.observation_space.low, self.observation_space.high)

    def _jains_index(self, x: np.ndarray) -> float:
        s = x.sum()
        if s == 0:
            return 1.0
        return float((s ** 2) / (self.n_ue * (x ** 2).sum() + 1e-9))

    def _compute_reward(self, throughputs: np.ndarray) -> float:
        norm_tp = throughputs.sum() / 1e6   # Mbps, unnormalised but consistent
        fairness = self._jains_index(throughputs)
        return self.alpha * norm_tp + self.beta * fairness
