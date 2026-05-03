import yaml
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Config:
    n_ue: int = 10
    n_rb: int = 50
    cell_radius: float = 500.0
    alpha: float = 0.5
    beta: float = 0.5
    lr: float = 5e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 50000
    buffer_size: int = 100000
    batch_size: int = 64
    target_update_freq: int = 1000
    hidden_size: int = 256
    n_episodes: int = 5000
    max_steps: int = 200
    checkpoint_every: int = 100


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(**{k: v for k, v in data.items() if k in Config.__dataclass_fields__})
