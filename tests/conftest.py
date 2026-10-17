import numpy as np
import pytest
import torch

from drl5g.config import DQNConfig, EnvConfig, ScenarioConfig, TrainConfig

torch.set_num_threads(1)


@pytest.fixture
def small_config() -> ScenarioConfig:
    """A cell small enough to train in about a second."""
    return ScenarioConfig(
        name="T",
        env=EnvConfig(n_ue=4, n_rb=20, max_steps=5, throughput_norm_mbps=20.0),
        dqn=DQNConfig(
            rbg_size=5, hidden_sizes=(32, 32), learning_starts=64, target_update_freq=100
        ),
        train=TrainConfig(episodes=400, seed=0),
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
