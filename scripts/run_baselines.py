"""
Run all baseline schedulers + DQN across all scenarios and write comparison.csv.

Each (scenario, scheduler) cell is evaluated across 5 seeds (0..4). The CSV
reports mean and std of per-seed average throughput / fairness.

Usage:
    python scripts/run_baselines.py
"""

import os
import sys
import csv
import numpy as np

# Allow imports from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from env.network_env import FiveGEnv
from baselines.schedulers import RandomScheduler, RoundRobinScheduler, ProportionalFairScheduler
from agents.dqn_scheduler import DQNScheduler
from evaluation.metrics import evaluate_scheduler

N_EPISODES = 50
SEEDS = [0, 1, 2, 3, 4]
OUT_PATH = os.path.join("results", "baselines", "comparison.csv")

CSV_HEADER = [
    "scenario", "scheduler",
    "mean_throughput_mbps", "std_throughput_mbps",
    "mean_fairness", "std_fairness",
    "n_seeds",
]


def load_cfg(scenario: str) -> dict:
    path = os.path.join("configs", f"scenario_{scenario}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def build_scheduler(name: str, n_ue: int, n_rb: int, scenario: str, seed: int):
    if name == "Random":
        return RandomScheduler(n_ue=n_ue, n_rb=n_rb, seed=seed)
    if name == "RoundRobin":
        return RoundRobinScheduler(n_ue=n_ue, n_rb=n_rb)
    if name == "ProportionalFair":
        return ProportionalFairScheduler(n_ue=n_ue, n_rb=n_rb)
    if name == "DQN":
        return DQNScheduler(n_ue=n_ue, n_rb=n_rb, scenario=scenario, seed=seed)
    raise ValueError(name)


def evaluate_cell(scheduler_name: str, scenario: str, cfg: dict) -> dict:
    n_ue, n_rb = cfg["n_ue"], cfg["n_rb"]
    tps, fairs = [], []
    for seed in SEEDS:
        sched = build_scheduler(scheduler_name, n_ue, n_rb, scenario, seed)
        env = FiveGEnv(
            n_ue=n_ue, n_rb=n_rb,
            cell_radius=cfg.get("cell_radius", 500.0),
            alpha=cfg.get("alpha", 0.5),
            beta=cfg.get("beta", 0.5),
            max_steps=cfg.get("max_steps", 200),
            seed=seed,
        )
        result = evaluate_scheduler(env, sched, n_episodes=N_EPISODES)
        tps.append(result["avg_throughput_mbps"])
        fairs.append(result["avg_fairness"])
    return {
        "mean_throughput_mbps": float(np.mean(tps)),
        "std_throughput_mbps":  float(np.std(tps)),
        "mean_fairness":        float(np.mean(fairs)),
        "std_fairness":         float(np.std(fairs)),
        "n_seeds":              len(SEEDS),
    }


def main():
    np.random.seed(0)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    scenarios = ["a", "b", "c", "d"]
    scheduler_names = ["Random", "RoundRobin", "ProportionalFair", "DQN"]

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        for scenario in scenarios:
            cfg = load_cfg(scenario)
            for sched_name in scheduler_names:
                agg = evaluate_cell(sched_name, scenario, cfg)
                writer.writerow([
                    scenario,
                    sched_name,
                    round(agg["mean_throughput_mbps"], 4),
                    round(agg["std_throughput_mbps"], 4),
                    round(agg["mean_fairness"], 4),
                    round(agg["std_fairness"], 4),
                    agg["n_seeds"],
                ])
                print(f"  scenario={scenario} {sched_name:18s} "
                      f"tp={agg['mean_throughput_mbps']:.2f}±{agg['std_throughput_mbps']:.2f} Mbps "
                      f"fair={agg['mean_fairness']:.3f}±{agg['std_fairness']:.3f}  "
                      f"(n_seeds={agg['n_seeds']})")

    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
