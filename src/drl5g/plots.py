"""Figures for the README and results write-up, drawn from the CSV outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
HIGHLIGHT = "#2a78d6"  # the DQN, the series every figure is about
CONTEXT = "#9a9993"  # baselines: identified by direct labels, not by colour
REFERENCE = "#eb6834"

SCHEDULER_ORDER = ("DQN", "Greedy", "PropFair", "RoundRobin", "Random", "MaxRate")
LABELS = {
    "DQN": "DQN (ours)",
    "Greedy": "Greedy (knows R)",
    "PropFair": "Proportional fair",
    "RoundRobin": "Round robin",
    "Random": "Random",
    "MaxRate": "Max C/I",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_MUTED,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
        }
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _by_scenario(summary: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in summary:
        grouped.setdefault(row["scenario"], {})[row["scheduler"]] = row
    return dict(sorted(grouped.items()))


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def objective_dotplot(summary_csv: Path, out: Path, titles: dict[str, str]) -> Path:
    """Mean per-TTI objective with 95 % CI, one panel per scenario."""
    _style()
    scenarios = _by_scenario(_read_csv(summary_csv))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.0 * len(scenarios), 3.0), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (scenario, rows) in zip(axes, scenarios.items(), strict=True):
        names = [n for n in SCHEDULER_ORDER if n in rows][::-1]
        y = np.arange(len(names))
        means = np.array([float(rows[n]["reward"]) for n in names])
        cis = np.array([float(rows[n]["reward_ci95"]) for n in names])
        colors = [HIGHLIGHT if n == "DQN" else CONTEXT for n in names]
        ax.hlines(y, means - cis, means + cis, colors=colors, linewidth=2)
        ax.scatter(means, y, s=42, c=colors, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.set_yticks(y, [LABELS[n] for n in names])
        ax.set_title(titles.get(scenario, scenario.upper()), loc="left")
        ax.set_xlabel("Objective R (mean per TTI)")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return _save(fig, out)


def tradeoff_scatter(summary_csv: Path, out: Path, titles: dict[str, str]) -> Path:
    """Throughput vs fairness per scheduler: where each one sits on the trade-off."""
    _style()
    scenarios = _by_scenario(_read_csv(summary_csv))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.2 * len(scenarios), 3.2), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (scenario, rows) in zip(axes, scenarios.items(), strict=True):
        for name in SCHEDULER_ORDER:
            if name not in rows:
                continue
            x, y = float(rows[name]["throughput_mbps"]), float(rows[name]["fairness"])
            is_dqn = name == "DQN"
            ax.scatter(
                x,
                y,
                s=60 if is_dqn else 40,
                c=HIGHLIGHT if is_dqn else CONTEXT,
                edgecolors=SURFACE,
                linewidths=1.5,
                zorder=3,
            )
            ax.annotate(
                LABELS[name],
                (x, y),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=7.5,
                color=TEXT if is_dqn else TEXT_MUTED,
                fontweight="bold" if is_dqn else "normal",
            )
        ax.set_title(titles.get(scenario, scenario.upper()), loc="left")
        ax.set_xlabel("Cell throughput (Mbit/s)")
        ax.set_ylim(0, 1.05)
        ax.margins(x=0.25)
    axes[0].set_ylabel("Jain fairness (per TTI)")
    fig.tight_layout()
    return _save(fig, out)


def training_curves(
    log_csvs: dict[str, Path],
    summary_csv: Path,
    out: Path,
    titles: dict[str, str],
) -> Path:
    """Validation objective of the greedy DQN policy during training.

    Horizontal lines mark the evaluated round-robin and greedy objectives of
    the same scenario, as reference levels.
    """
    _style()
    scenarios = _by_scenario(_read_csv(summary_csv))
    fig, axes = plt.subplots(1, len(log_csvs), figsize=(3.0 * len(log_csvs), 2.8))
    axes = np.atleast_1d(axes)

    for ax, (scenario, path) in zip(axes, sorted(log_csvs.items()), strict=True):
        rows = [r for r in _read_csv(path) if r["validation_objective"]]
        episodes = [int(r["episode"]) for r in rows]
        values = [float(r["validation_objective"]) for r in rows]
        ax.plot(episodes, values, color=HIGHLIGHT, linewidth=2, solid_capstyle="round")

        for name, color in (("RoundRobin", CONTEXT), ("Greedy", REFERENCE)):
            if name in scenarios.get(scenario, {}):
                level = float(scenarios[scenario][name]["reward"])
                ax.axhline(level, color=color, linewidth=1.2, linestyle=(0, (4, 3)))
                ax.annotate(
                    LABELS[name],
                    (1.0, level),
                    xycoords=("axes fraction", "data"),
                    xytext=(-2, 3),
                    textcoords="offset points",
                    ha="right",
                    fontsize=7,
                    color=TEXT_MUTED,
                )
        ax.set_title(titles.get(scenario, scenario.upper()), loc="left")
        ax.set_xlabel("Training episode (TTIs)")
    axes[0].set_ylabel("Validation objective R")
    fig.tight_layout()
    return _save(fig, out)
