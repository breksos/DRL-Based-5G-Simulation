"""Run a trained DQN checkpoint behind the common :class:`Scheduler` interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from drl5g.config import EnvConfig, ScenarioConfig, config_from_dict
from drl5g.rl.allocation import RBGAllocationTask
from drl5g.rl.dqn import EquivariantQNetwork
from drl5g.schedulers.base import LastInputCache

CHECKPOINT_FORMAT = 2


class DQNScheduler:
    """Greedy (ε = 0) roll-out of a trained Q-network, one RBG decision at a time.

    The channel is static within an episode, so the allocation for a given
    capacity vector never changes and is computed once per episode.
    """

    name = "DQN"

    def __init__(self, q_net: EquivariantQNetwork, env_config: EnvConfig, rbg_size: int) -> None:
        self._q_net = q_net.eval()
        self._task = RBGAllocationTask(env_config, rbg_size)
        self._cache = LastInputCache()

    @classmethod
    def from_checkpoint(cls, path: str | Path, env_config: EnvConfig | None = None) -> DQNScheduler:
        """Load a checkpoint written by :func:`drl5g.rl.trainer.save_checkpoint`.

        ``env_config`` overrides the cell the model was trained on, e.g. to run
        a scenario-A model on scenario D; the network itself is size-agnostic.
        """
        checkpoint = load_checkpoint(path)
        trained = config_from_dict(checkpoint["config"])
        q_net = EquivariantQNetwork(trained.dqn.hidden_sizes)
        q_net.load_state_dict(checkpoint["q_net"])
        return cls(q_net, env_config or trained.env, trained.dqn.rbg_size)

    def allocate(self, rb_capacity_bps: np.ndarray) -> np.ndarray:
        return self._cache.get_or_compute(rb_capacity_bps, self._greedy_rollout)

    @torch.no_grad()
    def _greedy_rollout(self, rb_capacity_bps: np.ndarray) -> np.ndarray:
        obs = self._task.reset(rb_capacity_bps)
        while not self._task.done:
            q_values = self._q_net(torch.from_numpy(obs).unsqueeze(0)).squeeze(0)
            obs, _, _ = self._task.step(int(q_values.argmax()))
        return self._task.allocation

    def reset(self) -> None:
        pass


def load_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path} is not a v{CHECKPOINT_FORMAT} drl5g checkpoint; "
            "models from the v1.0 course submission are not compatible"
        )
    return checkpoint


def trained_config(path: str | Path) -> ScenarioConfig:
    """The scenario configuration a checkpoint was trained with."""
    return config_from_dict(load_checkpoint(path)["config"])
