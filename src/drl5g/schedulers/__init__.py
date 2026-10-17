"""Scheduler implementations and a name-based factory."""

from __future__ import annotations

from pathlib import Path

from drl5g.config import ScenarioConfig
from drl5g.paths import model_path
from drl5g.schedulers.base import Scheduler
from drl5g.schedulers.classic import (
    GreedyObjectiveScheduler,
    MaxRateScheduler,
    ProportionalFairScheduler,
    RandomScheduler,
    RoundRobinScheduler,
)

SCHEDULER_NAMES = ("Random", "RoundRobin", "PropFair", "MaxRate", "Greedy", "DQN")

__all__ = [
    "SCHEDULER_NAMES",
    "DQNScheduler",
    "GreedyObjectiveScheduler",
    "MaxRateScheduler",
    "ProportionalFairScheduler",
    "RandomScheduler",
    "RoundRobinScheduler",
    "Scheduler",
    "make_scheduler",
]


def make_scheduler(
    name: str,
    config: ScenarioConfig,
    scenario: str,
    seed: int | None = None,
    checkpoint: str | Path | None = None,
) -> Scheduler:
    """Build a scheduler by name for the cell described by ``config``.

    The DQN loads ``checkpoint`` or, by default, the committed model of
    ``scenario``. A missing model is an error rather than a silent fallback.
    """
    env = config.env
    if name == "Random":
        return RandomScheduler(env.n_ue, env.n_rb, seed=seed)
    if name == "RoundRobin":
        return RoundRobinScheduler(env.n_ue, env.n_rb)
    if name == "PropFair":
        return ProportionalFairScheduler(env.n_ue, env.n_rb)
    if name == "MaxRate":
        return MaxRateScheduler(env.n_ue, env.n_rb)
    if name == "Greedy":
        return GreedyObjectiveScheduler(env, rbg_size=config.dqn.rbg_size)
    if name == "DQN":
        from drl5g.schedulers.dqn import DQNScheduler

        path = Path(checkpoint) if checkpoint else model_path(scenario)
        if not path.exists():
            raise FileNotFoundError(f"no trained model at {path}; run `drl5g train {scenario}`")
        return DQNScheduler.from_checkpoint(path, env_config=env)
    raise ValueError(f"unknown scheduler '{name}' (choose from {', '.join(SCHEDULER_NAMES)})")


def __getattr__(name: str):
    # The DQN scheduler pulls in PyTorch; import it only when it is used.
    if name == "DQNScheduler":
        from drl5g.schedulers.dqn import DQNScheduler

        return DQNScheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
