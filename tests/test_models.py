"""The committed models must match the committed configs and run on them."""

import numpy as np
import pytest

from drl5g.config import list_scenarios, load_scenario
from drl5g.env import FiveGEnv
from drl5g.paths import model_path
from drl5g.schedulers import DQNScheduler
from drl5g.schedulers.dqn import trained_config


@pytest.mark.parametrize("scenario", list_scenarios())
def test_committed_model_matches_its_scenario(scenario):
    path = model_path(scenario)
    if not path.exists():
        pytest.skip(f"no model committed for scenario {scenario}")
    assert trained_config(path) == load_scenario(scenario)

    config = load_scenario(scenario)
    env = FiveGEnv(config.env, seed=0)
    env.reset()
    allocation = DQNScheduler.from_checkpoint(path).allocate(env.rb_capacity_bps)
    assert allocation.shape == (config.env.n_rb,)
    assert allocation.min() >= 0 and allocation.max() < config.env.n_ue
    assert np.unique(allocation).size > 1
