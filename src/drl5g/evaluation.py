"""Run schedulers on the environment and aggregate the results.

Every scheduler is evaluated on the same UE drops: for a given seed, each one
gets a freshly seeded environment, so differences between schedulers are
paired (common random numbers) rather than confounded with luck of the drop.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from drl5g.config import ScenarioConfig
from drl5g.env import FiveGEnv
from drl5g.metrics import jain_index
from drl5g.schedulers import Scheduler, make_scheduler

Z_95 = 1.96
METRICS = ("reward", "throughput_mbps", "fairness", "fairness_longterm", "min_ue_mbps", "starved")


@dataclass(frozen=True)
class EpisodeResult:
    """Per-episode metrics, each averaged over the episode's TTIs.

    ``fairness`` is the mean per-TTI Jain index (what the reward sees);
    ``fairness_longterm`` is Jain over each UE's episode-average throughput.
    ``min_ue_mbps`` is the worst-served UE's average throughput and
    ``starved`` the fraction of UEs that received nothing all episode.
    """

    scenario: str
    scheduler: str
    seed: int
    episode: int
    reward: float
    throughput_mbps: float
    fairness: float
    fairness_longterm: float
    min_ue_mbps: float
    starved: float


def run_episode(env: FiveGEnv, scheduler: Scheduler) -> dict[str, float]:
    """Play one episode and return its metrics (without identifying fields)."""
    env.reset()
    scheduler.reset()
    rewards, fairness = [], []
    ue_throughput = np.zeros(env.n_ue)
    done = False
    while not done:
        allocation = scheduler.allocate(env.rb_capacity_bps)
        _, reward, terminated, truncated, info = env.step(allocation)
        rewards.append(reward)
        fairness.append(info["fairness"])
        ue_throughput += info["throughput_bps"]
        done = terminated or truncated

    ue_mean_mbps = ue_throughput / env.step_count / 1e6
    return {
        "reward": float(np.mean(rewards)),
        "throughput_mbps": float(ue_mean_mbps.sum()),
        "fairness": float(np.mean(fairness)),
        "fairness_longterm": jain_index(ue_mean_mbps),
        "min_ue_mbps": float(ue_mean_mbps.min()),
        "starved": float(np.mean(ue_mean_mbps == 0.0)),
    }


def evaluate_scenario(
    scenario: str,
    config: ScenarioConfig,
    scheduler_names: Iterable[str],
    seeds: Sequence[int],
    episodes: int,
    checkpoint: str | Path | None = None,
) -> list[EpisodeResult]:
    results = []
    for name in scheduler_names:
        for seed in seeds:
            env = FiveGEnv(config.env, seed=seed)
            scheduler = make_scheduler(name, config, scenario, seed=seed, checkpoint=checkpoint)
            for episode in range(episodes):
                metrics = run_episode(env, scheduler)
                results.append(EpisodeResult(scenario, name, seed, episode, **metrics))
    return results


def summarize(results: Iterable[EpisodeResult]) -> list[dict[str, float | str | int]]:
    """Mean and 95 % confidence half-width of every metric per (scenario, scheduler).

    UE drops are i.i.d., so each episode is one independent sample.
    """
    groups: dict[tuple[str, str], list[EpisodeResult]] = defaultdict(list)
    for result in results:
        groups[(result.scenario, result.scheduler)].append(result)

    rows = []
    for (scenario, scheduler), group in groups.items():
        row: dict[str, float | str | int] = {
            "scenario": scenario,
            "scheduler": scheduler,
            "episodes": len(group),
        }
        for metric in METRICS:
            values = np.array([getattr(r, metric) for r in group])
            row[metric] = round(float(values.mean()), 5)
            row[f"{metric}_ci95"] = round(_ci_half_width(values), 5)
        rows.append(row)
    return rows


def paired_difference(
    results: Iterable[EpisodeResult], scenario: str, a: str, b: str, metric: str = "reward"
) -> tuple[float, float]:
    """Mean and 95 % half-width of ``metric(a) − metric(b)`` over shared drops."""
    by_key = {
        (r.scheduler, r.seed, r.episode): getattr(r, metric)
        for r in results
        if r.scenario == scenario
    }
    diffs = np.array(
        [value - by_key[(b, seed, ep)] for (name, seed, ep), value in by_key.items() if name == a]
    )
    return float(diffs.mean()), _ci_half_width(diffs)


def _ci_half_width(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(Z_95 * values.std(ddof=1) / np.sqrt(values.size))


def write_rows(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_episodes(results: Sequence[EpisodeResult], path: Path) -> None:
    write_rows([asdict(r) for r in results], path)


def read_episodes(path: Path) -> list[EpisodeResult]:
    types = {f.name: f.type for f in fields(EpisodeResult)}
    casts = {"str": str, "int": int, "float": float}
    with open(path, encoding="utf-8") as f:
        return [
            EpisodeResult(**{k: casts[types[k]](v) for k, v in row.items()})
            for row in csv.DictReader(f)
        ]
