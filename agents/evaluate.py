"""
Eğitilmiş DQN modelini greedy policy (ε=0) ile değerlendirir.
Kullanım:
  python evaluate.py --model ../results/models/model_ep5000.pth --scenario a
  python evaluate.py --model ../results/models/model_ep5000.pth --scenario a --episodes 500
"""

import os
import sys
import csv
import argparse
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from dqn_agent import DQNAgent

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_env_and_adapter(cfg: dict):
    """FiveGEnv + ActionAdapter yükler. Berk'in env'i hazır değilse hata verir."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from env.network_env import FiveGEnv
    from train import ActionAdapter

    env = FiveGEnv(n_ue=cfg["n_ue"], n_rb=cfg["n_rb"],
                   alpha=cfg.get("alpha", 0.5), beta=cfg.get("beta", 0.5))
    adapter = ActionAdapter(n_ue=cfg["n_ue"], n_rb=cfg["n_rb"])

    class AdaptedEnv:
        def __init__(self, e, a): self._e = e; self._a = a
        def reset(self): return self._e.reset()
        def step(self, action): return self._e.step(self._a.to_allocation(action))

    return AdaptedEnv(env, adapter), adapter.action_dim


def evaluate(model_path: str, scenario: str, n_episodes: int = 500):
    cfg_path = os.path.join(CONFIGS_DIR, f"scenario_{scenario}.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    env, action_dim = load_env_and_adapter(cfg)
    state_dim = 2 * cfg["n_ue"] + cfg["n_rb"]

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    agent.load(model_path)
    agent.epsilon = 0.0  # greedy — kesinlikle en iyi bilinen aksiyonu seç

    episode_rewards = []
    print(f"=== Greedy Evaluation | Senaryo {scenario.upper()} | {n_episodes} episode ===")

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            ep_reward += reward

        episode_rewards.append(ep_reward)

        if episode % 100 == 0:
            print(f"  Episode {episode:4d} | Reward: {ep_reward:8.3f} | "
                  f"Avg: {np.mean(episode_rewards):8.3f}")

    avg = float(np.mean(episode_rewards))
    std = float(np.std(episode_rewards))
    print(f"\nSonuç → Avg: {avg:.4f}  Std: {std:.4f}")

    # Convergence curve CSV — Emre'nin grafik scripti bunu okuyacak
    out_dir = os.path.join(RESULTS_DIR, "logs")
    os.makedirs(out_dir, exist_ok=True)
    curve_path = os.path.join(out_dir, f"convergence_scenario_{scenario}.csv")
    with open(curve_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "avg_reward"])
        running_avg = []
        for i, r in enumerate(episode_rewards, 1):
            running_avg.append(r)
            writer.writerow([i, round(r, 4), round(float(np.mean(running_avg)), 4)])

    print(f"Convergence CSV kaydedildi → {curve_path}")
    return {"scenario": scenario, "avg_reward": avg, "std_reward": std}


def arch_experiment(n_episodes: int = 2000):
    """2×128 vs 3×256 network karşılaştırması. Mock env üzerinde çalışır."""
    from train import MockEnv

    configs = [
        {"tag": "2x128", "hidden_sizes": [128, 128]},
        {"tag": "3x256", "hidden_sizes": [256, 256, 256]},
    ]

    results = []
    for conf in configs:
        env = MockEnv()
        import time
        agent = DQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_sizes=conf["hidden_sizes"],
            epsilon_decay_steps=10_000,
        )

        rewards = []
        t0 = time.time()
        for episode in range(1, n_episodes + 1):
            state = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.store_transition(state, action, reward, next_state, done)
                agent.train_step()
                state = next_state
                ep_reward += reward
            rewards.append(ep_reward)

        elapsed = time.time() - t0
        final_avg = float(np.mean(rewards[-100:]))
        results.append({
            "tag": conf["tag"],
            "hidden_sizes": str(conf["hidden_sizes"]),
            "final_avg_reward": round(final_avg, 4),
            "train_time_sec": round(elapsed, 2),
        })
        print(f"[{conf['tag']}] Final avg: {final_avg:.2f} | Süre: {elapsed:.1f}s")

    out_path = os.path.join(RESULTS_DIR, "sweeps", "arch_experiment.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "hidden_sizes",
                                                "final_avg_reward", "train_time_sec"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nArch experiment tamamlandı → {out_path}")
    best = max(results, key=lambda x: x["final_avg_reward"])
    print(f"En iyi mimari: {best['tag']}  |  Final avg: {best['final_avg_reward']}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    # greedy evaluation
    p_eval = subparsers.add_parser("eval", help="Eğitilmiş modeli greedy ile değerlendir")
    p_eval.add_argument("--model", required=True, help="Model .pth dosyası")
    p_eval.add_argument("--scenario", required=True, choices=["a", "b", "c", "d"])
    p_eval.add_argument("--episodes", type=int, default=500)

    # arch experiment
    p_arch = subparsers.add_parser("arch", help="2x128 vs 3x256 network karşılaştırması")
    p_arch.add_argument("--episodes", type=int, default=2000)

    args = parser.parse_args()

    if args.cmd == "eval":
        evaluate(args.model, args.scenario, args.episodes)
    elif args.cmd == "arch":
        arch_experiment(args.episodes)
    else:
        parser.print_help()
