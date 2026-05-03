"""
Entry point.

Kullanım:
  python main.py --scenario a                        # Senaryo A eğit
  python main.py --scenario a --episodes 2000        # Kısa eğitim
  python main.py --demo                              # 10 adım random demo
  python main.py --gui                               # GUI aç
  python main.py --new --name e --n_ue 12 --n_rb 60  # Canlı yeni senaryo oluştur + eğit
"""

import argparse
import os
import yaml
import numpy as np
from pathlib import Path
from utils.config import load_config, Config
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


def create_scenario(name: str, n_ue: int, n_rb: int,
                    cell_radius: float = 500.0, n_episodes: int = 2000):
    """Yeni senaryo YAML'ı oluşturur ve Config döndürür."""
    cfg_path = Path("configs") / f"scenario_{name}.yaml"
    data = {
        "n_ue": n_ue, "n_rb": n_rb, "cell_radius": cell_radius,
        "alpha": 0.5, "beta": 0.5,
        "lr": 0.0005, "gamma": 0.99,
        "epsilon_start": 1.0, "epsilon_end": 0.01,
        "epsilon_decay_steps": min(50000, n_episodes * 100),
        "target_update_freq": 1000,
        "buffer_size": 100000, "batch_size": 64,
        "hidden_size": 256, "n_episodes": n_episodes,
        "max_steps": 200, "checkpoint_every": 100,
    }
    with open(cfg_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"Yeni senaryo oluşturuldu: {cfg_path}")
    print(f"  {n_ue} UE | {n_rb} RB | {n_episodes} episode | cell_radius={cell_radius}m")
    return load_config(cfg_path)


def run_training(scenario: str, cfg, n_episodes_override: int = None):
    from agents.train import train, ActionAdapter
    from agents.dqn_agent import DQNAgent

    env = FiveGEnv(n_ue=cfg.n_ue, n_rb=cfg.n_rb,
                   cell_radius=cfg.cell_radius,
                   alpha=cfg.alpha, beta=cfg.beta,
                   max_steps=cfg.max_steps)

    adapter   = ActionAdapter(n_ue=cfg.n_ue, n_rb=cfg.n_rb)
    state_dim = 2 * cfg.n_ue + cfg.n_rb
    n_ep      = n_episodes_override or cfg.n_episodes

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=adapter.action_dim,
        lr=cfg.lr, gamma=cfg.gamma,
        epsilon_decay_steps=cfg.epsilon_decay_steps,
        buffer_size=cfg.buffer_size, batch_size=cfg.batch_size,
        target_update_freq=cfg.target_update_freq,
        hidden_sizes=[cfg.hidden_size, cfg.hidden_size],
    )

    class AdaptedEnv:
        def reset(self_): return env.reset()
        def step(self_, a): return env.step(adapter.to_allocation(a))

    log_path       = os.path.join("results", "logs",   f"train_scenario_{scenario}.csv")
    checkpoint_dir = os.path.join("results", "models", f"scenario_{scenario}")

    print(f"State dim: {state_dim} | Action dim: {adapter.action_dim} | Episodes: {n_ep}")
    train(AdaptedEnv(), agent, n_episodes=n_ep,
          log_path=log_path, checkpoint_dir=checkpoint_dir,
          checkpoint_every=cfg.checkpoint_every)

    # Final model kaydet — GUI bunu otomatik bulur
    os.makedirs(checkpoint_dir, exist_ok=True)
    agent.save(os.path.join(checkpoint_dir, "model_final.pth"))
    env.close()
    print(f"\nModel kaydedildi → results/models/scenario_{scenario}/model_final.pth")
    print("GUI'de görüntülemek için: python main.py --gui")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="a",
                        help="Senaryo adı: a, b, c, d veya özel isim")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Episode sayısı override")
    parser.add_argument("--demo",  action="store_true", help="10 adım random demo")
    parser.add_argument("--gui",   action="store_true", help="GUI aç")
    parser.add_argument("--new",   action="store_true", help="Yeni senaryo oluştur")
    parser.add_argument("--name",  type=str, default=None, help="Yeni senaryo adı (--new ile)")
    parser.add_argument("--n_ue",  type=int, default=10,   help="UE sayısı (--new ile)")
    parser.add_argument("--n_rb",  type=int, default=50,   help="RB sayısı (--new ile)")
    parser.add_argument("--radius",type=float, default=500.0, help="Hücre yarıçapı m (--new ile)")
    args = parser.parse_args()

    if args.gui:
        from gui.app import main as gui_main
        gui_main()
        return

    if args.demo:
        demo()
        return

    if args.new:
        name = args.name or args.scenario
        n_ep = args.episodes or 2000
        cfg  = create_scenario(name, args.n_ue, args.n_rb, args.radius, n_ep)
        run_training(name, cfg)
        return

    # Normal eğitim
    cfg_path = Path("configs") / f"scenario_{args.scenario}.yaml"
    if not cfg_path.exists():
        print(f"HATA: {cfg_path} bulunamadı.")
        print("İpucu: Yeni senaryo için --new --name e --n_ue 12 --n_rb 60 kullan")
        return

    cfg = load_config(cfg_path)
    print(f"Config yüklendi: {cfg_path}")
    print(f"  n_ue={cfg.n_ue}, n_rb={cfg.n_rb}, episodes={args.episodes or cfg.n_episodes}")
    run_training(args.scenario, cfg, args.episodes)


if __name__ == "__main__":
    main()
