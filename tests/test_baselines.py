import sys
import os
import csv
import numpy as np
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import baselines.schedulers as schedulers_mod
from baselines.schedulers import RandomScheduler, RoundRobinScheduler, ProportionalFairScheduler
from env.network_env import FiveGEnv
from evaluation.metrics import evaluate_scheduler, jains_fairness

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


def test_pf_per_rb_distributes_across_ues():
    """Per-RB PF must average per-step Jain > 1/n_ue (the strawman PF ceiling)."""
    n_ue, n_rb = 4, 8
    env = FiveGEnv(n_ue=n_ue, n_rb=n_rb, seed=0)
    pf = ProportionalFairScheduler(n_ue=n_ue, n_rb=n_rb)
    state, _ = env.reset(seed=0)
    pf.reset()

    fairness_per_step = []
    for _ in range(50):
        action = pf.allocate(state)
        state, _, _, _, info = env.step(action)
        fairness_per_step.append(info["fairness"])

    avg_fair = float(np.mean(fairness_per_step))
    assert avg_fair > 0.25, (
        f"per-RB PF should average per-step Jain > 0.25, got {avg_fair:.3f}"
    )


def test_pf_avg_rate_updates_only_chosen_ues_per_rb():
    """One allocate() with n_rb=2: only chosen UEs' avg_rates change."""
    pf = ProportionalFairScheduler(n_ue=3, n_rb=2)
    pf._avg_rates = np.array([1.0, 1.0, 1000.0], dtype=np.float64)
    initial_avg = pf._avg_rates.copy()

    fake_instant = np.array([2.0, 1.999, 0.001], dtype=np.float64)
    with patch.object(schedulers_mod, "capacity_bps", return_value=fake_instant):
        state = np.zeros(2 * 3 + 2, dtype=np.float32)
        action = pf.allocate(state)

    assert action[0] == 0, f"expected RB0→UE0, got {action[0]}"
    assert action[1] == 1, f"expected RB1→UE1, got {action[1]}"
    assert pf._avg_rates[2] == initial_avg[2], (
        f"UE2 was not chosen — avg_rate must be unchanged "
        f"(was {initial_avg[2]}, now {pf._avg_rates[2]})"
    )
    assert pf._avg_rates[0] != initial_avg[0], "UE0 chosen — avg_rate should have moved"
    assert pf._avg_rates[1] != initial_avg[1], "UE1 chosen — avg_rate should have moved"


def test_run_baselines_csv_schema():
    """comparison.csv header must match the multi-seed schema."""
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "baselines", "comparison.csv"
    )
    if not os.path.exists(csv_path):
        pytest.skip("comparison.csv not yet generated")
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    expected = [
        "scenario", "scheduler",
        "mean_throughput_mbps", "std_throughput_mbps",
        "mean_fairness", "std_fairness", "n_seeds",
    ]
    assert header == expected, f"CSV header mismatch: {header}"
