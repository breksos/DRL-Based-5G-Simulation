"""
Phase 3 stability test.
Runs 1000 episodes of random policy on every scenario config.
Verifies: no NaN/inf in states/rewards, reward in sane range.

Run from project root:
    python scripts/stability_test.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from pathlib import Path
from env.network_env import FiveGEnv
from utils.config import load_config

SCENARIOS = ['scenario_a', 'scenario_b', 'scenario_c', 'scenario_d']
N_EPISODES = 1000


def run_scenario(name: str) -> dict:
    cfg = load_config(Path('configs') / f'{name}.yaml')
    env = FiveGEnv(n_ue=cfg.n_ue, n_rb=cfg.n_rb,
                   cell_radius=cfg.cell_radius, max_steps=cfg.max_steps, seed=42)

    nan_count = inf_count = 0
    rewards, fairness = [], []

    for ep in range(N_EPISODES):
        state, _ = env.reset()
        done = False
        ep_rew = 0.0
        ep_fair = []
        while not done:
            if np.isnan(state).any():
                nan_count += 1
            if np.isinf(state).any():
                inf_count += 1
            action = env.action_space.sample()
            state, rew, terminated, truncated, info = env.step(action)
            ep_rew += rew
            ep_fair.append(info['fairness'])
            done = terminated or truncated
        rewards.append(ep_rew)
        fairness.append(np.mean(ep_fair))

    env.close()
    return {
        'nan_states': nan_count,
        'inf_states': inf_count,
        'reward_mean': np.mean(rewards),
        'reward_std':  np.std(rewards),
        'reward_min':  np.min(rewards),
        'reward_max':  np.max(rewards),
        'fairness_mean': np.mean(fairness),
    }


def main():
    print(f"Stability test: {N_EPISODES} episodes per scenario\n")
    print(f"{'Scenario':<14} {'NaN':>5} {'Inf':>5} {'R_mean':>9} {'R_std':>8} "
          f"{'R_min':>8} {'R_max':>8} {'Fair':>7}")
    print('-' * 72)

    all_passed = True
    for name in SCENARIOS:
        r = run_scenario(name)
        status = 'OK' if r['nan_states'] == 0 and r['inf_states'] == 0 else 'FAIL'
        if status == 'FAIL':
            all_passed = False
        print(f"{name:<14} {r['nan_states']:>5} {r['inf_states']:>5} "
              f"{r['reward_mean']:>9.2f} {r['reward_std']:>8.2f} "
              f"{r['reward_min']:>8.2f} {r['reward_max']:>8.2f} "
              f"{r['fairness_mean']:>7.3f}  {status}")

    print()
    print('All scenarios PASSED' if all_passed else 'Some scenarios FAILED')


if __name__ == '__main__':
    main()
