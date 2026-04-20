"""
Entry point.

Usage:
  python main.py --scenario a          # train DQN on scenario A
  python main.py --scenario b --eval   # evaluate saved model
  python main.py --demo                # run 10 steps with random policy and print results
"""

import argparse
import numpy as np
from pathlib import Path
from utils.config import load_config
from env.network_env import FiveGEnv


def demo():
    env = FiveGEnv(n_ue=10, n_rb=50, seed=0)
    state, _ = env.reset()
    print(f"{'Step':>4}  {'Reward':>8}  {'Throughput (Mbps)':>18}  {'Fairness':>9}")
    print("-" * 50)
    for step in range(10):
        action = env.action_space.sample()
        state, reward, _, _, info = env.step(action)
        tp = info["throughput_total"] / 1e6
        print(f"{step+1:>4}  {reward:>8.3f}  {tp:>18.2f}  {info['fairness']:>9.4f}")
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="a", choices=["a","b","c","d"],
                        help="Which scenario config to use")
    parser.add_argument("--eval", action="store_true", help="Evaluate saved model")
    parser.add_argument("--demo", action="store_true", help="10-step random demo")
    parser.add_argument("--gui",  action="store_true", help="Launch simulation GUI")
    args = parser.parse_args()

    if args.gui:
        from gui.app import main as gui_main
        gui_main()
        return

    if args.demo:
        demo()
        return

    cfg_path = Path("configs") / f"scenario_{args.scenario}.yaml"
    cfg = load_config(cfg_path)
    print(f"Loaded config: {cfg_path}")
    print(f"  n_ue={cfg.n_ue}, n_rb={cfg.n_rb}, episodes={cfg.n_episodes}")

    env = FiveGEnv(n_ue=cfg.n_ue, n_rb=cfg.n_rb,
                   cell_radius=cfg.cell_radius,
                   alpha=cfg.alpha, beta=cfg.beta,
                   max_steps=cfg.max_steps)

    print("Environment ready. State dim:", env.observation_space.shape[0])
    print("(Training loop: plug in agents/train.py)")
    env.close()


if __name__ == "__main__":
    main()
