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
CONTEXT = "#9a9993"  # baselines in the dot plot, identified by their axis labels
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


# Max C/I leaves 90 %+ of the users unserved, so its objective is around −0.4;
# drawing it would squash the differences that matter into a sliver.
OMITTED = ("MaxRate",)
OMITTED_NOTE = (
    "Max C/I is not shown: it leaves 90–95 % of users unserved (R ≈ −0.42). "
    "See results/evaluation/summary.csv."
)

# Scatter identity: palette colour plus marker shape, so overlapping points stay legible.
SCATTER_STYLE = {
    "DQN": {"color": HIGHLIGHT, "marker": "o", "s": 70},
    "Greedy": {"color": "#eb6834", "marker": "D", "s": 42},
    "PropFair": {"facecolors": "none", "edgecolors": "#1baf7a", "marker": "o", "s": 150},
    "RoundRobin": {"color": "#eda100", "marker": "s", "s": 36},
    "Random": {"color": "#e87ba4", "marker": "v", "s": 46},
}


def _plotted(rows: dict[str, dict[str, str]]) -> list[str]:
    return [n for n in SCHEDULER_ORDER if n in rows and n not in OMITTED]


def objective_dotplot(summary_csv: Path, out: Path, titles: dict[str, str]) -> Path:
    """Mean per-TTI objective with 95 % CI, one panel per scenario."""
    _style()
    scenarios = _by_scenario(_read_csv(summary_csv))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.0 * len(scenarios), 2.9), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (scenario, rows) in zip(axes, scenarios.items(), strict=True):
        names = _plotted(rows)[::-1]
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
        ax.margins(x=0.12)
    fig.text(0.01, -0.04, OMITTED_NOTE, fontsize=7.5, color=TEXT_MUTED)
    fig.tight_layout()
    return _save(fig, out)


def tradeoff_scatter(summary_csv: Path, out: Path, titles: dict[str, str]) -> Path:
    """Throughput vs fairness per scheduler: where each one sits on the trade-off."""
    _style()
    scenarios = _by_scenario(_read_csv(summary_csv))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.2 * len(scenarios), 3.3))
    axes = np.atleast_1d(axes)

    for ax, (scenario, rows) in zip(axes, scenarios.items(), strict=True):
        for name in reversed(_plotted(rows)):  # DQN drawn last, on top
            style = {"edgecolors": SURFACE, "linewidths": 1.2, **SCATTER_STYLE[name]}
            if "facecolors" in SCATTER_STYLE[name]:
                style["linewidths"] = 1.8
            ax.scatter(
                float(rows[name]["throughput_mbps"]),
                float(rows[name]["fairness"]),
                label=LABELS[name],
                zorder=4 if name == "DQN" else 3,
                **style,
            )
        ax.set_title(titles.get(scenario, scenario.upper()), loc="left")
        ax.set_xlabel("Cell throughput (Mbit/s)")
        ax.margins(x=0.25, y=0.25)
    axes[0].set_ylabel("Jain fairness (per TTI)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[::-1],
        labels[::-1],
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
        fontsize=8.5,
    )
    fig.text(0.01, -0.04, OMITTED_NOTE, fontsize=7.5, color=TEXT_MUTED)
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

        # Greedy sits above round robin, so label it above its line and RR below.
        for name, color, offset, va in (
            ("RoundRobin", CONTEXT, -3, "top"),
            ("Greedy", REFERENCE, 3, "bottom"),
        ):
            if name in scenarios.get(scenario, {}):
                level = float(scenarios[scenario][name]["reward"])
                ax.axhline(level, color=color, linewidth=1.2, linestyle=(0, (4, 3)))
                ax.annotate(
                    LABELS[name],
                    (1.0, level),
                    xycoords=("axes fraction", "data"),
                    xytext=(-2, offset),
                    textcoords="offset points",
                    ha="right",
                    va=va,
                    fontsize=7,
                    color=TEXT_MUTED,
                )
        ax.set_title(titles.get(scenario, scenario.upper()), loc="left")
        ax.set_xlabel("Training episode (TTIs)")
    axes[0].set_ylabel("Validation objective R")
    fig.tight_layout()
    return _save(fig, out)
