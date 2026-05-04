"""
Tum baseline scheduler'lari (RR, PF, Random) ve DQN'i tum senaryolarda degerlendirir.

Calistir (repo kokundan):
    python evaluation/evaluate_baselines.py

Cikti:
    results/baselines/scenario_{a,b,c,d}.csv  -- per-scenario sonuclar
    results/baselines/summary.csv             -- tum sonuclar tek CSV
    results/baselines/per_ue_{scenario}.csv   -- per-episode per-UE throughput (boxplot icin)
"""
import os
import sys
import csv
import yaml
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env.network_env import FiveGEnv
from baselines.schedulers import RoundRobinScheduler, ProportionalFairScheduler, RandomScheduler
from agents.dqn_scheduler import DQNScheduler
from evaluation.metrics import evaluate_scheduler, per_ue_throughput

SCENARIOS        = ["a", "b", "c", "d"]
CONFIGS_DIR      = os.path.join(ROOT, "configs")
RESULTS_DIR      = os.path.join(ROOT, "results", "baselines")
N_EPISODES       = 1000   # aggregate stats icin
N_EPISODES_PER_UE = 100   # per-UE boxplot verisi icin (daha hizli)


def load_cfg(scenario):
    with open(os.path.join(CONFIGS_DIR, f"scenario_{scenario}.yaml")) as f:
        return yaml.safe_load(f)


def collect_per_ue(env, scheduler, n_ue, n_episodes=100):
    """
    Her episode icin per-UE toplam throughput (Mbps) toplar.
    Donus: np.array shape (n_episodes * n_ue,) -- duzlestirilen per-episode degerler.
    Her deger: bir episodeda o UE'nin aldigi toplam throughput (Mbps cinsinden).
    """
    all_ep_tp = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        scheduler.reset()
        done = False
        ep_ue_tp = np.zeros(n_ue, dtype=np.float64)
        while not done:
            snrs   = state[:n_ue]
            action = scheduler.allocate(state)
            state, _, terminated, truncated, _ = env.step(action)
            ep_ue_tp += per_ue_throughput(action, snrs) / 1e6   # Mbps per step
            done = terminated or truncated
        all_ep_tp.append(ep_ue_tp.copy())
    return np.array(all_ep_tp).flatten()   # shape: (n_episodes * n_ue,)


def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_rows = []

    for scenario in SCENARIOS:
        print(f"\n=== Senaryo {scenario.upper()} ===")
        cfg   = load_cfg(scenario)
        n_ue  = cfg["n_ue"]
        n_rb  = cfg["n_rb"]

        env = FiveGEnv(
            n_ue=n_ue, n_rb=n_rb,
            cell_radius=cfg.get("cell_radius", 500.0),
            alpha=cfg.get("alpha", 0.5),
            beta=cfg.get("beta", 0.5),
            max_steps=cfg.get("max_steps", 200),
        )

        schedulers = {
            "RR":     RoundRobinScheduler(n_ue, n_rb),
            "PF":     ProportionalFairScheduler(n_ue, n_rb),
            "Random": RandomScheduler(n_ue, n_rb),
            "DQN":    DQNScheduler(n_ue, n_rb, scenario=scenario),
        }

        scenario_rows = []
        per_ue_data   = {}

        for name, sched in schedulers.items():
            if name == "DQN" and not sched.is_loaded:
                print(f"  [{name}] Model bulunamadi, atlandi.")
                continue

            print(f"  [{name}] {N_EPISODES} episode (aggregate)...")
            result = evaluate_scheduler(env, sched, n_episodes=N_EPISODES)

            row = {
                "scenario":            scenario,
                "scheduler":           name,
                "n_ue":                n_ue,
                "n_rb":                n_rb,
                "avg_throughput_mbps": round(result["avg_throughput_mbps"], 4),
                "std_throughput_mbps": round(result["std_throughput_mbps"], 4),
                "avg_fairness":        round(result["avg_fairness"], 4),
                "std_fairness":        round(result["std_fairness"], 4),
            }
            scenario_rows.append(row)
            summary_rows.append(row)
            print(f"         TP: {result['avg_throughput_mbps']:.2f} +/- "
                  f"{result['std_throughput_mbps']:.2f} Mbps | "
                  f"Fairness: {result['avg_fairness']:.4f}")

            print(f"  [{name}] {N_EPISODES_PER_UE} episode (per-UE)...")
            per_ue_data[name] = collect_per_ue(env, sched, n_ue, N_EPISODES_PER_UE)

        if not scenario_rows:
            env.close()
            continue

        # Per-scenario aggregate CSV
        sc_path = os.path.join(RESULTS_DIR, f"scenario_{scenario}.csv")
        with open(sc_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(scenario_rows[0].keys()))
            writer.writeheader()
            writer.writerows(scenario_rows)
        print(f"  -> {sc_path}")

        # Per-UE CSV (wide format: sutun = yontem, satir = veri noktasi)
        pu_path      = os.path.join(RESULTS_DIR, f"per_ue_{scenario}.csv")
        method_names = list(per_ue_data.keys())
        max_len      = max(len(v) for v in per_ue_data.values())
        with open(pu_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(method_names)
            for i in range(max_len):
                row = [
                    round(float(per_ue_data[m][i]), 6) if i < len(per_ue_data[m]) else ""
                    for m in method_names
                ]
                writer.writerow(row)
        print(f"  -> {pu_path}")

        env.close()

    # Master summary CSV
    if summary_rows:
        summary_path = os.path.join(RESULTS_DIR, "summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nOzet: {summary_path}")
        print(f"Toplam {len(summary_rows)} satir ({len(SCENARIOS)} senaryo x scheduler).")


if __name__ == "__main__":
    run_all()
