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
# 5G ortamında action space N_UE^N_RB boyutunda — DQN için çok büyük.
# Çözüm: DQN sadece N_UE adet skor üretir (hangi UE'ye öncelik verilmeli).
# Adapter bu skorları alıp her RB'yi en yüksek skorlu UE'ye atar.

class ActionAdapter:
    def __init__(self, n_ue: int, n_rb: int):
        self.n_ue = n_ue
        self.n_rb = n_rb
        # DQN'in gördüğü action sayısı: N_UE
        self.action_dim = n_ue

    def to_allocation(self, dqn_action: int) -> np.ndarray:
        """DQN'den gelen tek integer action'ı N_RB boyutlu allocation array'e çevirir.

        dqn_action: hangi UE'nin bu adımda öncelikli olduğu (0..N_UE-1).
        Strateji: seçilen UE'ye RB'lerin yarısını ver, kalanı round-robin dağıt.
        Bu sayede agent "kime öncelik ver" kararını öğrenir.
        """
        allocation = np.zeros(self.n_rb, dtype=np.int64)
        half = self.n_rb // 2
        # İlk yarı: öncelikli UE
        allocation[:half] = dqn_action
        # İkinci yarı: round-robin
        for i in range(half, self.n_rb):
            allocation[i] = i % self.n_ue
        return allocation


# ---------------------------------------------------------------------------
# Mock Environment (CartPole benzeri) — gymnasium olmadan çalışır
# ---------------------------------------------------------------------------

class MockEnv:
    """4 boyutlu state, 2 action (sol/sağ). Reward = 1.0 her adım.
    200 adımda done. DQN'in temel öğrenme döngüsünü doğrulamak için kullanılır.
    """
    def __init__(self):
        self.state_dim = 4
        self.action_dim = 2
        self._step = 0

    def reset(self):
        self._step = 0
        return np.random.uniform(-0.05, 0.05, size=4).astype(np.float32)

    def step(self, action):
        self._step += 1
        next_state = np.random.uniform(-0.05, 0.05, size=4).astype(np.float32)
        reward = 1.0
        done = self._step >= 200
        return next_state, reward, done, False, {}


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
# Eğitim Döngüsü
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

            # Reward normalizasyonu — Q-değerlerini küçük tutar, loss spike'larını önler
            agent.store_transition(state, action, reward / 100.0, next_state, done)
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

    # Final model kaydet — GUI ve DQNScheduler bunu kullanır
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_path = os.path.join(checkpoint_dir, "model_final.pth")
        agent.save(final_path)
        print(f"Final model kaydedildi: {final_path}")

    print("Eğitim tamamlandı.")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "5g"], default="mock",
                        help="mock: CartPole benzeri test | 5g: gerçek ortam (Phase 3'te)")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--log", type=str, default="../results/logs/train_log.csv")
    parser.add_argument("--checkpoint_dir", type=str, default="../results/models")
    args = parser.parse_args()

    if args.mode == "mock":
        print("=== Mock Environment Testi ===")
        env = MockEnv()
        agent = DQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            lr=args.lr,
            epsilon_decay_steps=10_000,  # mock için daha hızlı decay
        )
        train(env, agent, n_episodes=args.episodes,
              log_path=args.log, checkpoint_dir=None)

    elif args.mode == "5g":
        # Phase 3'te Berk'in FiveGEnv'i hazır olduğunda burası aktif edilir
        import sys
        sys.path.append("..")
        from env.network_env import FiveGEnv
        import yaml

        with open("../configs/default.yaml") as f:
            cfg = yaml.safe_load(f)

        env = FiveGEnv(n_ue=cfg["n_ue"], n_rb=cfg["n_rb"])
        adapter = ActionAdapter(n_ue=cfg["n_ue"], n_rb=cfg["n_rb"])

        # State dim: 2*N_UE + N_RB (SNR + throughput + rb_occupancy)
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

        # 5G ortamında env.step() allocation array bekler, agent integer döndürür
        # Bu yüzden env.step'i adapter ile sarıyoruz
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
