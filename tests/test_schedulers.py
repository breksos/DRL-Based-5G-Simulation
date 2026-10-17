import numpy as np
import pytest

from drl5g.channel import capacity_bps, drop_ues, snr_db
from drl5g.config import EnvConfig, ScenarioConfig
from drl5g.schedulers import (
    GreedyObjectiveScheduler,
    MaxRateScheduler,
    ProportionalFairScheduler,
    RandomScheduler,
    RoundRobinScheduler,
    make_scheduler,
)

N_UE, N_RB = 20, 50


@pytest.fixture
def capacity() -> np.ndarray:
    distances, _ = drop_ues(N_UE, 500.0, np.random.default_rng(0))
    return capacity_bps(snr_db(distances))


@pytest.mark.parametrize("name", ["Random", "RoundRobin", "PropFair", "MaxRate", "Greedy"])
def test_every_scheduler_returns_a_valid_allocation(name, capacity):
    config = ScenarioConfig(name="T", env=EnvConfig(n_ue=N_UE, n_rb=N_RB))
    scheduler = make_scheduler(name, config, scenario="t", seed=0)
    scheduler.reset()
    allocation = scheduler.allocate(capacity)
    assert allocation.shape == (N_RB,)
    assert np.issubdtype(allocation.dtype, np.integer)
    assert allocation.min() >= 0 and allocation.max() < N_UE


def test_unknown_scheduler_is_rejected():
    with pytest.raises(ValueError, match="unknown scheduler"):
        make_scheduler("Oracle", ScenarioConfig(name="T"), scenario="t")


def test_missing_dqn_model_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_scheduler("DQN", ScenarioConfig(name="T"), "t", checkpoint=tmp_path / "none.pth")


def test_random_is_reproducible(capacity):
    a = RandomScheduler(N_UE, N_RB, seed=5).allocate(capacity)
    b = RandomScheduler(N_UE, N_RB, seed=5).allocate(capacity)
    np.testing.assert_array_equal(a, b)


def test_round_robin_continues_where_it_stopped(capacity):
    rr = RoundRobinScheduler(N_UE, N_RB)
    counts = sum(np.bincount(rr.allocate(capacity), minlength=N_UE) for _ in range(2))
    np.testing.assert_array_equal(counts, 5)  # 2 TTIs × 50 RBs over 20 UEs
    rr.reset()
    assert rr.allocate(capacity)[0] == 0


def test_max_rate_serves_only_the_best_ue(capacity):
    allocation = MaxRateScheduler(N_UE, N_RB).allocate(capacity)
    np.testing.assert_array_equal(allocation, np.argmax(capacity))


def test_proportional_fair_starves_nobody_and_equalises_rbs(capacity):
    pf = ProportionalFairScheduler(N_UE, N_RB)
    counts = sum(np.bincount(pf.allocate(capacity), minlength=N_UE) for _ in range(200))
    assert counts.min() > 0
    # Static channel: long-run PF gives every UE the same number of RBs.
    np.testing.assert_allclose(counts / counts.sum(), 1 / N_UE, atol=0.01)


def test_greedy_starts_with_the_best_channel(capacity):
    greedy = GreedyObjectiveScheduler(EnvConfig(n_ue=N_UE, n_rb=N_RB), rbg_size=5)
    allocation = greedy.allocate(capacity)
    assert np.all(allocation[:5] == np.argmax(capacity))
    np.testing.assert_array_equal(allocation, greedy.allocate(capacity))
