import os
import csv
import argparse
import collections
import numpy as np

try:
    from agents.dqn_agent import DQNAgent
except ImportError:
    from dqn_agent import DQNAgent


# ---------------------------------------------------------------------------
# Action Space Adapter
# ---------------------------------------------------------------------------
# Action space is N_UE^N_RB — too large for DQN.
# Solution: DQN outputs one score per UE (priority signal).
# Adapter assigns all RBs to the chosen UE (pure greedy).
# This gives the DQN full control over all RBs while keeping action_dim = N_UE.

class ActionAdapter:
    def __init__(self, n_ue: int, n_rb: int):
        self.n_ue = n_ue
        self.n_rb = n_rb
        self.action_dim = n_ue

    def to_allocation(self, dqn_action: int) -> np.ndarray:
        """Map a single UE index to an N_RB allocation (all RBs to that UE)."""
        return np.full(self.n_rb, dqn_action, dtype=np.int64)


# ---------------------------------------------------------------------------
# CSV Logger
# ---------------------------------------------------------------------------

class CSVLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["episode", "episode_reward", "avg_reward_100", "epsilon", "loss"])

    def log(self, episode, episode_reward, avg_reward_100, epsilon, loss):
        self.writer.writerow([episode, round(episode_reward, 4),
                               round(avg_reward_100, 4), round(epsilon, 5), round(loss, 6)])
        self.file.flush()

    def close(self):
        self.file.close()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(env, agent: DQNAgent, n_episodes: int, log_path: str,
          checkpoint_dir: str = None, checkpoint_every: int = 100):

    logger = CSVLogger(log_path)
    reward_window = collections.deque(maxlen=100)

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        episode_reward = 0.0
        last_loss = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            result = env.step(action)
            next_state, reward, terminated, truncated, _ = result
            done = terminated or truncated

            # Reward is already normalised in env._compute_reward; store directly.
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss:
                last_loss = loss

            state = next_state
            episode_reward += reward

        reward_window.append(episode_reward)
        avg_100 = np.mean(reward_window)

        logger.log(episode, episode_reward, avg_100, agent.epsilon, last_loss)

        if episode % 50 == 0:
            print(f"Episode {episode:5d} | Reward: {episode_reward:7.1f} | "
                  f"Avg100: {avg_100:7.1f} | eps: {agent.epsilon:.4f} | Loss: {last_loss:.5f}")

        if checkpoint_dir and episode % checkpoint_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            agent.save(os.path.join(checkpoint_dir, f"model_ep{episode}.pth"))

    logger.close()

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_path = os.path.join(checkpoint_dir, "model_final.pth")
        agent.save(final_path)
        print(f"Final model saved: {final_path}")

    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["5g"], default="5g")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--log", type=str, default="../results/logs/train_log.csv")
    parser.add_argument("--checkpoint_dir", type=str, default="../results/models")
    args = parser.parse_args()

    import sys
    sys.path.append("..")
    from env.network_env import FiveGEnv
    import yaml

    with open("../configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    env = FiveGEnv(n_ue=cfg["n_ue"], n_rb=cfg["n_rb"])
    adapter = ActionAdapter(n_ue=cfg["n_ue"], n_rb=cfg["n_rb"])

    state_dim = 2 * cfg["n_ue"] + cfg["n_rb"]

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=adapter.action_dim,
        lr=args.lr,
        gamma=cfg.get("gamma", 0.99),
        epsilon_decay_steps=cfg.get("epsilon_decay_steps", 50_000),
        buffer_size=cfg.get("buffer_size", 100_000),
        batch_size=cfg.get("batch_size", 64),
    )

    class AdaptedEnv:
        def __init__(self, env, adapter):
            self._env = env
            self._adapter = adapter

        def reset(self):
            return self._env.reset()

        def step(self, action: int):
            allocation = self._adapter.to_allocation(action)
            return self._env.step(allocation)

    adapted_env = AdaptedEnv(env, adapter)
    train(adapted_env, agent, n_episodes=args.episodes,
          log_path=args.log, checkpoint_dir=args.checkpoint_dir)
