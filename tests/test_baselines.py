import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from baselines.schedulers import RandomScheduler, RoundRobinScheduler, ProportionalFairScheduler
from env.network_env import FiveGEnv
from evaluation.metrics import evaluate_scheduler

_STATE_70 = np.zeros(70, dtype=np.float32)
_STATE_70[:10] = 10.0


def test_round_robin_rotates():
    rr = RoundRobinScheduler(n_ue=10, n_rb=50)
    a0 = rr.allocate(_STATE_70)
    a1 = rr.allocate(_STATE_70)
    a2 = rr.allocate(_STATE_70)
    assert a0[0] == 0
    assert a1[0] == 1
    assert a2[0] == 2


def test_pf_no_initial_bias():
    pf = ProportionalFairScheduler(n_ue=10, n_rb=50)
    assert pf._avg_rates is None
    pf.allocate(_STATE_70)
    assert pf._avg_rates is not None
    pf.reset()
    assert pf._avg_rates is None


def test_random_seeded():
    r1 = RandomScheduler(n_ue=10, n_rb=50, seed=0)
    r2 = RandomScheduler(n_ue=10, n_rb=50, seed=0)
    assert np.array_equal(r1.allocate(_STATE_70), r2.allocate(_STATE_70))


def test_evaluate_scheduler_finite():
    env = FiveGEnv(n_ue=10, n_rb=50, seed=0)
    sched = RandomScheduler(n_ue=10, n_rb=50, seed=42)
    result = evaluate_scheduler(env, sched, n_episodes=2)
    assert np.isfinite(result["avg_throughput_mbps"])
    assert np.isfinite(result["avg_fairness"])


def test_reward_is_clamped():
    env = FiveGEnv(n_ue=10, n_rb=50, seed=0)
    env.reset()
    action = np.zeros(50, dtype=int)
    _, reward, _, _, _ = env.step(action)
    assert reward <= 1.0, f"reward {reward} exceeds 1.0 — norm_tp not clamped"
    env.close()
