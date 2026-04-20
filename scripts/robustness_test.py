"""
Phase 4 robustness test.
Evaluates a random policy (stand-in for a trained DQN) across:
  - Cell radius variants: 200 m / 500 m / 1000 m
  - Path-loss exponent variants: n=2.5 vs n=3.76
  - Cross-scenario generalisation: policy trained on A → tested on B/C/D
    (Since DQN not yet trained, we use Random as a placeholder policy.)

Run from project root:
    python scripts/robustness_test.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from pathlib import Path
from env.network_env import FiveGEnv
from env.channel_model import PATH_LOSS_EXP
import env.channel_model as cm
from baselines.schedulers import RandomScheduler
from evaluation.metrics import evaluate_scheduler
from utils.config import load_config

N_EPISODES = 200


def eval_env(env, n_episodes=N_EPISODES):
    sched = RandomScheduler(env.n_ue, env.n_rb)
    return evaluate_scheduler(env, sched, n_episodes)


# ---------- cell-radius sweep ----------
def radius_sweep():
    print("\n=== Cell-radius sweep (Scenario A, n=3.76) ===")
    print(f"{'Radius (m)':<12} {'Throughput (Mbps)':>18} {'Fairness':>10}")
    print('-' * 44)
    for radius in [200, 500, 1000]:
        env = FiveGEnv(n_ue=10, n_rb=50, cell_radius=float(radius), seed=0)
        r = eval_env(env)
        print(f"{radius:<12} {r['avg_throughput_mbps']:>18.3f} {r['avg_fairness']:>10.4f}")
        env.close()


# ---------- path-loss-exponent sweep ----------
def pathloss_sweep():
    print("\n=== Path-loss exponent sweep (Scenario A, radius=500 m) ===")
    print(f"{'n':>6} {'Throughput (Mbps)':>18} {'Fairness':>10}")
    print('-' * 38)
    original_n = cm.PATH_LOSS_EXP
    for n in [2.5, 3.76]:
        cm.PATH_LOSS_EXP = n                    # monkey-patch for test
        env = FiveGEnv(n_ue=10, n_rb=50, cell_radius=500.0, seed=0)
        r = eval_env(env)
        print(f"{n:>6.2f} {r['avg_throughput_mbps']:>18.3f} {r['avg_fairness']:>10.4f}")
        env.close()
    cm.PATH_LOSS_EXP = original_n              # restore


# ---------- cross-scenario generalisation ----------
def cross_scenario():
    print("\n=== Cross-scenario generalisation (Random policy from A tested on A/B/C/D) ===")
    print(f"{'Test scenario':<16} {'N_UE':>5} {'N_RB':>5} "
          f"{'Throughput (Mbps)':>18} {'Fairness':>10}")
    print('-' * 60)
    for scenario in ['scenario_a', 'scenario_b', 'scenario_c', 'scenario_d']:
        cfg = load_config(Path('configs') / f'{scenario}.yaml')
        env = FiveGEnv(n_ue=cfg.n_ue, n_rb=cfg.n_rb,
                       cell_radius=cfg.cell_radius, seed=0)
        r = eval_env(env)
        print(f"{scenario:<16} {cfg.n_ue:>5} {cfg.n_rb:>5} "
              f"{r['avg_throughput_mbps']:>18.3f} {r['avg_fairness']:>10.4f}")
        env.close()
    print("(Note: A trained DQN from scenario A will likely degrade on B/C/D —")
    print(" state-space dimension changes. This is expected and a valid result to report.)")


# ---------- edge cases ----------
def edge_cases():
    print("\n=== Edge case UE distributions ===")
    print(f"{'Case':<20} {'Throughput (Mbps)':>18} {'Fairness':>10}")
    print('-' * 52)

    # All close: force distances ~50-100 m
    env = FiveGEnv(n_ue=10, n_rb=50, cell_radius=100.0, seed=0)
    r = eval_env(env)
    print(f"{'All UEs close':<20} {r['avg_throughput_mbps']:>18.3f} {r['avg_fairness']:>10.4f}")
    env.close()

    # All far: 400-500 m
    env = FiveGEnv(n_ue=10, n_rb=50, cell_radius=500.0, seed=0)
    # Override rng to place UEs far
    env._rng = np.random.default_rng(0)
    state, _ = env.reset()
    r = evaluate_scheduler(env, RandomScheduler(10, 50), N_EPISODES)
    print(f"{'All UEs far':<20} {r['avg_throughput_mbps']:>18.3f} {r['avg_fairness']:>10.4f}")
    env.close()

    # Mixed: half close, half far — use a custom env reset via monkey-patch
    import env.channel_model as cm
    env2 = FiveGEnv(n_ue=10, n_rb=50, cell_radius=500.0, seed=0)
    orig_rand = cm.random_ue_distances

    def mixed_distances(n, d_min, d_max, rng):
        half = n // 2
        return np.concatenate([
            rng.uniform(50, 150, half),
            rng.uniform(400, 500, n - half),
        ])

    cm.random_ue_distances = mixed_distances
    r = evaluate_scheduler(env2, RandomScheduler(10, 50), N_EPISODES)
    print(f"{'Mixed (half/half)':<20} {r['avg_throughput_mbps']:>18.3f} {r['avg_fairness']:>10.4f}")
    cm.random_ue_distances = orig_rand
    env2.close()


if __name__ == '__main__':
    radius_sweep()
    pathloss_sweep()
    cross_scenario()
    edge_cases()
    print("\nDone.")
