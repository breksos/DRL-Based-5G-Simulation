import numpy as np
import pytest
import torch

from drl5g.config import EnvConfig
from drl5g.metrics import per_ue_throughput, scheduling_objective
from drl5g.rl.allocation import FEATURES_PER_UE, GLOBAL_FEATURES, RBGAllocationTask
from drl5g.rl.dqn import EpsilonSchedule, EquivariantQNetwork, ReplayBuffer


@pytest.fixture
def task(rng) -> RBGAllocationTask:
    return RBGAllocationTask(
        EnvConfig(n_ue=6, n_rb=30, starvation_penalty=1.0), rbg_size=5, rng=rng
    )


def test_rewards_telescope_to_the_final_objective(task, rng):
    task.reset()
    total = 0.0
    while not task.done:
        _, reward, _ = task.step(int(rng.integers(task.n_ue)))
        total += reward
    cfg = task.env_config
    throughput = per_ue_throughput(task.allocation, task._capacity)
    expected = scheduling_objective(
        throughput, cfg.alpha, cfg.beta, cfg.throughput_norm_mbps, cfg.starvation_penalty
    )
    assert task.objective == pytest.approx(expected)
    assert total == pytest.approx(expected - (-1.0))  # R(empty) = −λ when everyone starves


def test_task_shapes_and_termination(task):
    obs = task.reset()
    assert obs.shape == (task.observation_size,) == (4 * 6 + 4,)
    assert obs[-1] == 1.0 and obs[-2] == 1.0  # nothing allocated, everyone unserved
    for _ in range(task.n_decisions):
        obs, _, done = task.step(0)
    assert done and obs[-1] == 0.0
    assert task.allocation.shape == (30,)
    with pytest.raises(RuntimeError):
        task.step(0)


def test_q_network_is_permutation_equivariant(rng):
    torch.manual_seed(0)
    net = EquivariantQNetwork((16, 16))
    n_ue, k, g = 7, FEATURES_PER_UE, GLOBAL_FEATURES
    obs = rng.normal(size=k * n_ue + g).astype(np.float32)
    perm = rng.permutation(n_ue)
    permuted = np.concatenate([obs[:-g].reshape(k, n_ue)[:, perm].ravel(), obs[-g:]])
    q = net(torch.from_numpy(obs)[None])[0]
    q_perm = net(torch.from_numpy(permuted)[None])[0]
    torch.testing.assert_close(q_perm, q[perm])


def test_q_network_handles_any_number_of_ues():
    net = EquivariantQNetwork((8,))
    for n_ue in (3, 20):
        obs = torch.zeros(2, FEATURES_PER_UE * n_ue + GLOBAL_FEATURES)
        assert net(obs).shape == (2, n_ue)


def test_replay_buffer_wraps_around(rng):
    buffer = ReplayBuffer(capacity=3, obs_size=2, rng=rng)
    for i in range(5):
        buffer.add(np.full(2, i), i, float(i), np.full(2, i + 1), done=False)
    assert len(buffer) == 3
    _, actions, rewards, _, _ = buffer.sample(64)
    assert set(actions.tolist()) <= {2, 3, 4}
    torch.testing.assert_close(rewards, actions.float())


def test_epsilon_schedule():
    schedule = EpsilonSchedule(start=1.0, end=0.1, decay_steps=10)
    assert schedule(0) == 1.0
    assert schedule(5) == pytest.approx(0.55)
    assert schedule(10) == schedule(1000) == pytest.approx(0.1)
