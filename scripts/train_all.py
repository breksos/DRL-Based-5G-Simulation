"""
Tüm senaryoları sırayla eğitir.
Sunum öncesi bir kez çalıştır:
    python scripts/train_all.py
    python scripts/train_all.py --episodes 3000   # hızlı test için
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
from pathlib import Path
from utils.config import load_config
from env.network_env import FiveGEnv
from agents.dqn_agent import DQNAgent
from agents.train import train, ActionAdapter

SCENARIOS = ["a", "b", "c", "d"]


def train_scenario(scenario: str, episodes: int = None):
    cfg_path = Path("configs") / f"scenario_{scenario}.yaml"
    cfg = load_config(cfg_path)
    n_ep = episodes or cfg.n_episodes

    print(f"\n{'='*55}")
    print(f"  SENARYO {scenario.upper()}  |  {cfg.n_ue} UE  {cfg.n_rb} RB  |  {n_ep} episode")
    print(f"{'='*55}")

    env = FiveGEnv(n_ue=cfg.n_ue, n_rb=cfg.n_rb,
                   cell_radius=cfg.cell_radius,
                   alpha=cfg.alpha, beta=cfg.beta,
                   max_steps=cfg.max_steps)

    adapter   = ActionAdapter(n_ue=cfg.n_ue, n_rb=cfg.n_rb)
    state_dim = 2 * cfg.n_ue + cfg.n_rb

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=adapter.action_dim,
        lr=cfg.lr,
        gamma=cfg.gamma,
        epsilon_decay_steps=cfg.epsilon_decay_steps,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        target_update_freq=cfg.target_update_freq,
        hidden_sizes=[cfg.hidden_size, cfg.hidden_size],
    )

    class AdaptedEnv:
        def reset(self_): return env.reset()
        def step(self_, a): return env.step(adapter.to_allocation(a))

    log_path       = os.path.join("results", "logs",   f"train_scenario_{scenario}.csv")
    checkpoint_dir = os.path.join("results", "models", f"scenario_{scenario}")

    t0 = time.time()
    train(AdaptedEnv(), agent, n_episodes=n_ep,
          log_path=log_path, checkpoint_dir=checkpoint_dir,
          checkpoint_every=cfg.checkpoint_every)
    elapsed = time.time() - t0

    # Son modeli scenario_X_final.pth olarak da kaydet
    final_path = os.path.join(checkpoint_dir, "model_final.pth")
    agent.save(final_path)
    env.close()
    print(f"  Süre: {elapsed/60:.1f} dk  |  Final model: {final_path}")
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS,
                        choices=SCENARIOS + ["all"],
                        help="Hangi senaryolar eğitilsin (varsayılan: hepsi)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Episode sayısı override (varsayılan: config'deki değer)")
    args = parser.parse_args()

    targets = SCENARIOS if "all" in args.scenarios else args.scenarios
    total_t0 = time.time()

    for s in targets:
        train_scenario(s, args.episodes)

    total = time.time() - total_t0
    print(f"\n{'='*55}")
    print(f"  Tüm eğitimler tamamlandı — toplam {total/60:.1f} dk")
    print(f"{'='*55}")
