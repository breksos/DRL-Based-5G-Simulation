"""Default locations of configs and results, so commands work from any directory."""

from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path:
    """``$DRL5G_ROOT`` if set, else the source checkout, else the working directory."""
    if "DRL5G_ROOT" in os.environ:
        return Path(os.environ["DRL5G_ROOT"])
    checkout = Path(__file__).resolve().parents[2]
    return checkout if (checkout / "configs").is_dir() else Path.cwd()


PROJECT_ROOT = _project_root()
CONFIGS_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
LOGS_DIR = RESULTS_DIR / "logs"
EVALUATION_DIR = RESULTS_DIR / "evaluation"
FIGURES_DIR = RESULTS_DIR / "figures"


def model_path(scenario: str, models_dir: Path = MODELS_DIR) -> Path:
    return models_dir / f"scenario_{scenario}" / "model_final.pth"


def training_log_path(scenario: str, logs_dir: Path = LOGS_DIR) -> Path:
    return logs_dir / f"train_scenario_{scenario}.csv"
