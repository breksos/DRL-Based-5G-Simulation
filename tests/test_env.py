import numpy as np
import pytest

from drl5g.config import EnvConfig
from drl5g.env import FiveGEnv
from drl5g.metrics import per_ue_throughput, scheduling_objective


@pytest.fixture
def env() -> FiveGEnv:
    return FiveGEnv(EnvConfig(n_ue=6, n_rb=12, max_steps=3), seed=7)


def test_observation_layout(env):
    obs, _ = env.reset()
    assert obs.shape == (12,) and obs.dtype == np.float32
    np.testing.assert_allclose(obs[:6], env.snr_db, rtol=1e-5)
    np.testing.assert_array_equal(obs[6:], 0.0)
    assert env.observation_space.contains(obs)


def test_seeded_reset_is_reproducible(env):
    env.reset(seed=3)
    first = env.distances.copy()
    env.reset()
    env.reset(seed=3)
    np.testing.assert_array_equal(env.distances, first)


def test_step_reports_objective_and_throughput(env):
    env.reset()
    action = np.arange(12) % 6
    obs, reward, terminated, truncated, info = env.step(action)
    expected_tp = per_ue_throughput(action, env.rb_capacity_bps)
    np.testing.assert_allclose(info["throughput_bps"], expected_tp)
    cfg = env.config
    assert reward == pytest.approx(
        scheduling_objective(expected_tp, cfg.alpha, cfg.beta, cfg.throughput_norm_mbps)
    )
    np.testing.assert_allclose(obs[6:], expected_tp / 1e6, rtol=1e-5)
    assert not terminated and not truncated


def test_episode_truncates_after_max_steps(env):
    env.reset()
    flags = [env.step(np.zeros(12, dtype=int))[3] for _ in range(3)]
    assert flags == [False, False, True]
