"""Scenario configuration: one YAML file per scenario, loaded into typed dataclasses.

A scenario file has three sections::

    name: A
    description: Baseline load
    env:   {n_ue: 10, n_rb: 50, ...}     # -> EnvConfig
    dqn:   {lr: 0.0005, ...}             # -> DQNConfig
    train: {episodes: 3000, seed: 0}     # -> TrainConfig

Unknown keys are rejected so a typo cannot silently fall back to a default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from drl5g.channel import MIN_DISTANCE_M
from drl5g.paths import CONFIGS_DIR


@dataclass(frozen=True)
class EnvConfig:
    """Cell layout, episode length and the reward that defines "good scheduling"."""

    n_ue: int = 10
    n_rb: int = 50
    cell_radius: float = 500.0
    max_steps: int = 200
    alpha: float = 0.5
    beta: float = 0.5
    throughput_norm_mbps: float = 50.0
    starvation_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.n_ue < 1 or self.n_rb < 1:
            raise ValueError("n_ue and n_rb must be positive")
        if self.cell_radius <= MIN_DISTANCE_M:
            raise ValueError(f"cell_radius must exceed {MIN_DISTANCE_M} m")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")


@dataclass(frozen=True)
class DQNConfig:
    """Hyper-parameters of the Double-DQN scheduler."""

    rbg_size: int = 5
    hidden_sizes: tuple[int, ...] = (256, 256)
    lr: float = 5e-4
    gamma: float = 1.0
    batch_size: int = 64
    buffer_size: int = 100_000
    learning_starts: int = 1_000
    target_update_freq: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay_fraction: float = 0.6
    grad_clip_norm: float = 10.0


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 3_000
    seed: int = 0


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    description: str = ""
    env: EnvConfig = field(default_factory=EnvConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        if self.env.n_rb % self.dqn.rbg_size:
            raise ValueError(
                f"n_rb={self.env.n_rb} is not a multiple of rbg_size={self.dqn.rbg_size}"
            )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dqn"]["hidden_sizes"] = list(self.dqn.hidden_sizes)
        return data


def _build(cls: type, data: dict[str, Any] | None, section: str) -> Any:
    data = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown key(s) in '{section}': {', '.join(unknown)}")
    if "hidden_sizes" in data:
        data["hidden_sizes"] = tuple(data["hidden_sizes"])
    return cls(**data)


def config_from_dict(data: dict[str, Any]) -> ScenarioConfig:
    top_level = {"name", "description", "env", "dqn", "train"}
    unknown = sorted(set(data) - top_level)
    if unknown:
        raise ValueError(f"unknown top-level key(s): {', '.join(unknown)}")
    if "name" not in data:
        raise ValueError("scenario config needs a 'name'")
    return ScenarioConfig(
        name=str(data["name"]),
        description=str(data.get("description", "")),
        env=_build(EnvConfig, data.get("env"), "env"),
        dqn=_build(DQNConfig, data.get("dqn"), "dqn"),
        train=_build(TrainConfig, data.get("train"), "train"),
    )


def scenario_path(scenario: str, configs_dir: Path = CONFIGS_DIR) -> Path:
    return configs_dir / f"scenario_{scenario.lower()}.yaml"


def load_config(path: str | Path) -> ScenarioConfig:
    with open(path, encoding="utf-8") as f:
        return config_from_dict(yaml.safe_load(f) or {})


def load_scenario(scenario: str, configs_dir: Path = CONFIGS_DIR) -> ScenarioConfig:
    """Load ``configs/scenario_<id>.yaml`` by its short id, e.g. ``"a"``."""
    path = scenario_path(scenario, configs_dir)
    if not path.exists():
        available = ", ".join(list_scenarios(configs_dir)) or "none"
        raise FileNotFoundError(
            f"no scenario '{scenario}' in {configs_dir} (available: {available})"
        )
    return load_config(path)


def save_config(config: ScenarioConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False, allow_unicode=True)


def list_scenarios(configs_dir: Path = CONFIGS_DIR) -> list[str]:
    """Short ids of every ``scenario_*.yaml`` file, sorted."""
    return sorted(p.stem.removeprefix("scenario_") for p in configs_dir.glob("scenario_*.yaml"))
