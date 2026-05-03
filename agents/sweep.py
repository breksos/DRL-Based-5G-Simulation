"""
Hiperparametre sweep scripti.
Kullanım:
  python sweep.py --mode lr        → 3 farklı learning rate dener
  python sweep.py --mode reward    → 3 farklı alpha/beta kombinasyonu dener (5G env gerekir)
"""

import os
import sys
import csv
import argparse
import collections
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from dqn_agent import DQNAgent

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "sweeps")
CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


# ---------------------------------------------------------------------------
# Yardımcı: tek bir konfigürasyonla eğitim çalıştır, özet döndür
# ---------------------------------------------------------------------------

def run_experiment(env, state_dim: int, action_dim: int, cfg: dict,
                   n_episodes: int, tag: str) -> dict:
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=cfg["lr"],
        gamma=cfg.get("gamma", 0.99),
        epsilon_decay_steps=cfg.get("epsilon_decay_steps", 50_000),
        buffer_size=cfg.get("buffer_size", 100_000),
        batch_size=cfg.get("batch_size", 64),
    )

    reward_window = collections.deque(maxlen=100)
    episode_rewards = []

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            result = env.step(action)
            next_state, reward, terminated, truncated, _ = result
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            ep_reward += reward

        reward_window.append(ep_reward)
        episode_rewards.append(ep_reward)

        if episode % 100 == 0:
            print(f"  [{tag}] ep {episode:5d} | avg100: {np.mean(reward_window):7.2f} | ε: {agent.epsilon:.4f}")

    # Convergence: son 100 episode ortalaması
    final_avg = float(np.mean(episode_rewards[-100:]))
    # Convergence noktası: ilk kez son 100 ortalamasının %90'ına ulaştığı episode
    threshold = final_avg * 0.90
    convergence_ep = n_episodes
    running = collections.deque(maxlen=100)
    for i, r in enumerate(episode_rewards):
        running.append(r)
        if len(running) == 100 and np.mean(running) >= threshold:
            convergence_ep = i + 1
            break

    return {
        "tag": tag,
        "lr": cfg["lr"],
        "final_avg_reward": round(final_avg, 4),
        "convergence_episode": convergence_ep,
    }


# ---------------------------------------------------------------------------
# LR Sweep
# ---------------------------------------------------------------------------

def lr_sweep(n_episodes: int = 2000):
    from train import MockEnv
    env = MockEnv()
    state_dim = env.state_dim
    action_dim = env.action_dim

    with open(os.path.join(CONFIGS_DIR, "default.yaml")) as f:
        base_cfg = yaml.safe_load(f)

    lr_values = [1e-3, 5e-4, 1e-4]
    results = []

    print("=== LR Sweep Başladı ===")
    for lr in lr_values:
        cfg = {**base_cfg, "lr": lr}
        tag = f"lr={lr}"
        print(f"\n--- {tag} ---")
        summary = run_experiment(env, state_dim, action_dim, cfg, n_episodes, tag)
        results.append(summary)

    # Sonuçları CSV'ye kaydet
    out_path = os.path.join(RESULTS_DIR, "lr_sweep.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "lr", "final_avg_reward", "convergence_episode"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n=== LR Sweep Tamamlandı → {out_path} ===")
    best = max(results, key=lambda x: x["final_avg_reward"])
    print(f"En iyi LR: {best['lr']}  |  Final avg reward: {best['final_avg_reward']}")
    return results


# ---------------------------------------------------------------------------
# Reward Weight Sweep (5G env gerekir — Berk'in Phase 2 çıktısı)
# ---------------------------------------------------------------------------

def reward_sweep(n_episodes: int = 5000):
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from env.network_env import FiveGEnv
        from train import ActionAdapter
    except ImportError:
        print("HATA: FiveGEnv bulunamadı. Bu sweep Berk'in env'i hazır olduğunda çalışır.")
        return

    with open(os.path.join(CONFIGS_DIR, "default.yaml")) as f:
        base_cfg = yaml.safe_load(f)

    combos = [
        {"alpha": 0.7, "beta": 0.3, "tag": "a0.7_b0.3"},
        {"alpha": 0.5, "beta": 0.5, "tag": "a0.5_b0.5"},
        {"alpha": 0.3, "beta": 0.7, "tag": "a0.3_b0.7"},
    ]

    adapter = ActionAdapter(n_ue=base_cfg["n_ue"], n_rb=base_cfg["n_rb"])
    state_dim = 2 * base_cfg["n_ue"] + base_cfg["n_rb"]
    action_dim = adapter.action_dim
    results = []

    print("=== Reward Weight Sweep Başladı ===")
    for combo in combos:
        env = FiveGEnv(
            n_ue=base_cfg["n_ue"],
            n_rb=base_cfg["n_rb"],
            alpha=combo["alpha"],
            beta=combo["beta"],
        )

        class AdaptedEnv:
            def __init__(self, e, a): self._e = e; self._a = a
            def reset(self): return self._e.reset()
            def step(self, action): return self._e.step(self._a.to_allocation(action))

        cfg = {**base_cfg, "lr": base_cfg["lr"]}
        tag = combo["tag"]
        print(f"\n--- {tag} ---")
        summary = run_experiment(AdaptedEnv(env, adapter), state_dim, action_dim,
                                 cfg, n_episodes, tag)
        summary["alpha"] = combo["alpha"]
        summary["beta"] = combo["beta"]
        results.append(summary)

    out_path = os.path.join(RESULTS_DIR, "reward_sweep.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "alpha", "beta", "lr",
                                                "final_avg_reward", "convergence_episode"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n=== Reward Sweep Tamamlandı → {out_path} ===")
    return results


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["lr", "reward"], required=True)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Episode sayısı (varsayılan: lr=2000, reward=5000)")
    args = parser.parse_args()

    if args.mode == "lr":
        lr_sweep(n_episodes=args.episodes or 2000)
    elif args.mode == "reward":
        reward_sweep(n_episodes=args.episodes or 5000)
