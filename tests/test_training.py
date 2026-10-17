import dataclasses
import itertools

import numpy as np

from drl5g.channel import capacity_bps, drop_ues, snr_db
from drl5g.env import objective_for
from drl5g.metrics import per_ue_throughput
from drl5g.rl.trainer import train
from drl5g.schedulers import DQNScheduler, RoundRobinScheduler
from drl5g.schedulers.dqn import trained_config


def _test_drops(config, n=64):
    rng = np.random.default_rng(99)
    env = config.env
    return [capacity_bps(snr_db(drop_ues(env.n_ue, env.cell_radius, rng)[0])) for _ in range(n)]


def _mean_objective(scheduler, config):
    drops = _test_drops(config)
    return np.mean(
        [objective_for(config.env, per_ue_throughput(scheduler.allocate(c), c)) for c in drops]
    )


def _brute_force_optimum(config):
    """Best objective over every RBG allocation; feasible only for tiny cells."""
    env, rbg = config.env, config.dqn.rbg_size
    choices = list(itertools.product(range(env.n_ue), repeat=env.n_rb // rbg))
    return np.mean(
        [
            max(objective_for(env, per_ue_throughput(np.repeat(c, rbg), cap)) for c in choices)
            for cap in _test_drops(config)
        ]
    )


def test_training_writes_artifacts_and_nearly_reaches_the_optimum(tmp_path, small_config):
    result = train(small_config, tmp_path / "model.pth", tmp_path / "log.csv")
    assert result.checkpoint.exists() and result.log.exists()
    assert trained_config(result.checkpoint) == small_config

    dqn = _mean_objective(DQNScheduler.from_checkpoint(result.checkpoint), small_config)
    round_robin = _mean_objective(
        RoundRobinScheduler(small_config.env.n_ue, small_config.env.n_rb), small_config
    )
    assert dqn > round_robin
    assert dqn >= 0.97 * _brute_force_optimum(small_config)


def test_training_is_deterministic(tmp_path, small_config):
    config = dataclasses.replace(
        small_config, train=dataclasses.replace(small_config.train, episodes=60)
    )
    train(config, tmp_path / "a.pth", tmp_path / "a.csv")
    train(config, tmp_path / "b.pth", tmp_path / "b.csv")
    assert (tmp_path / "a.csv").read_text() == (tmp_path / "b.csv").read_text()
