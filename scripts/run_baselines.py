"""
Run all baseline schedulers + DQN across all scenarios and write comparison.csv.

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
OUT_PATH = os.path.join("results", "baselines", "comparison.csv")


def load_cfg(scenario: str) -> dict:
    path = os.path.join("configs", f"scenario_{scenario}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    np.random.seed(0)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    scenarios = ["a", "b", "c", "d"]

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario", "scheduler",
            "avg_throughput_mbps", "std_throughput_mbps",
            "avg_fairness", "std_fairness",
        ])

        for scenario in scenarios:
            cfg = load_cfg(scenario)
            n_ue, n_rb = cfg["n_ue"], cfg["n_rb"]

            schedulers = [
                ("Random",          RandomScheduler(n_ue=n_ue, n_rb=n_rb, seed=0)),
                ("RoundRobin",      RoundRobinScheduler(n_ue=n_ue, n_rb=n_rb)),
                ("ProportionalFair",ProportionalFairScheduler(n_ue=n_ue, n_rb=n_rb)),
                ("DQN",             DQNScheduler(n_ue=n_ue, n_rb=n_rb, scenario=scenario, seed=0)),
            ]

            for sched_name, sched in schedulers:
                env = FiveGEnv(
                    n_ue=n_ue, n_rb=n_rb,
                    cell_radius=cfg.get("cell_radius", 500.0),
                    alpha=cfg.get("alpha", 0.5),
                    beta=cfg.get("beta", 0.5),
                    max_steps=cfg.get("max_steps", 200),
                    seed=0,
                )
                results = evaluate_scheduler(env, sched, n_episodes=N_EPISODES)
                writer.writerow([
                    scenario,
                    sched_name,
                    round(results["avg_throughput_mbps"], 4),
                    round(results["std_throughput_mbps"], 4),
                    round(results["avg_fairness"], 4),
                    round(results["std_fairness"], 4),
                ])
                print(f"  scenario={scenario} {sched_name:20s} "
                      f"tp={results['avg_throughput_mbps']:.2f} Mbps "
                      f"fair={results['avg_fairness']:.3f}")

    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
