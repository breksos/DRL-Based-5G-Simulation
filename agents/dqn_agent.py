"""DQN agent bileşenleri: QNetwork, ReplayBuffer, DQNAgent."""

import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Tam bağlı Q-fonksiyonu yaklaşıklayıcısı.

    Args:
        state_dim: Giriş state vektörünün boyutu.
        action_dim: Çıkış (Q-değeri) sayısı — action space boyutu.
        hidden_sizes: Gizli katman boyutlarının listesi.
            Varsayılan [256, 256] plan gereksinimini karşılar;
            mimari deney için [128, 128] veya [256, 256, 256] geçilebilir.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_sizes: list = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        layers = []
        in_size = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        layers.append(nn.Linear(in_size, action_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization — gradyan patlaması/sönmesini önler."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Her action için Q-değeri döndürür. Shape: (batch, action_dim)."""
        return self.net(x)


class ReplayBuffer:
    """Experience Replay tamponu.

    Ardışık geçişler arasındaki korelasyonu kırmak için geçmiş deneyimler
    burada biriktirilir ve eğitimde rastgele mini-batch olarak örneklenir.
    Bu olmadan network ardışık örnekleri ezberler, genelleşemez.

    Args:
        maxlen: Tampon kapasitesi. Dolunca en eski kayıtlar silinir.
    """

    def __init__(self, maxlen: int = 100_000):
        self.buffer = collections.deque(maxlen=maxlen)

    def store(self, state, action, reward, next_state, done):
        """Tek bir geçişi tampona ekler.

        Args:
            state: Mevcut durum (np.ndarray).
            action: Alınan aksiyon (int).
            reward: Alınan ödül (float).
            next_state: Sonraki durum (np.ndarray).
            done: Episode bitti mi (bool).
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int = 64):
        """Tampondan rastgele mini-batch örnekler.

        Returns:
            Tuple: (states, actions, rewards, next_states, dones) —
            hepsi float32/int64 numpy array.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent.

    Vanilla DQN: experience replay + target network.
    Dueling DQN veya Double DQN gibi uzantılar future work kapsamında.

    Args:
        state_dim: State vektörü boyutu.
        action_dim: Discrete action sayısı.
        lr: Adam optimizer learning rate. 5e-4 sweep sonucunda en kararlı
            convergence'ı verdiği için default seçildi.
        gamma: Discount factor. 0.99 — agent uzun vadeli throughput/fairness
            dengesini öğrenmeli, sıfıra yakın değerler bunu engeller.
        epsilon_start: Başlangıç exploration oranı (tam rastgele).
        epsilon_end: Minimum exploration oranı. 0.01 — eğitim sonunda
            hâlâ %1 keşif yapılır, lokal minimuma takılmayı önler.
        epsilon_decay_steps: ε'nin epsilon_start'tan epsilon_end'e
            lineer düştüğü adım sayısı. 50k — buffer dolmadan önce
            yeterli keşif sağlanır.
        target_update_freq: Target network kaç adımda bir güncellenir.
            1000 — çok sık güncelleme moving-target sorununu geri getirir,
            çok seyrek güncelleme öğrenmeyi yavaşlatır.
        batch_size: Mini-batch boyutu.
        buffer_size: ReplayBuffer kapasitesi.
        hidden_sizes: QNetwork gizli katman boyutları.
        device: 'cuda' veya 'cpu'. None ise otomatik seçilir.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 50_000,
        target_update_freq: int = 1_000,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        hidden_sizes: list = None,
        device: str = None,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        hs = hidden_sizes or [256, 256]
        self.q_net = QNetwork(state_dim, action_dim, hidden_sizes=hs).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_sizes=hs).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(maxlen=buffer_size)

        self.step_count = 0
        self.last_loss = 0.0

    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy aksiyon seçimi.

        Args:
            state: Mevcut state vektörü.

        Returns:
            Seçilen aksiyon indeksi (int).
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q_net(state_t).argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        """Geçişi replay buffer'a kaydeder."""
        self.buffer.store(state, action, reward, next_state, done)

    def train_step(self) -> float:
        """Tek eğitim adımı: buffer'dan sample → Bellman hedefi → MSE → backprop.

        Buffer yeterince dolmadan (< batch_size) çağrılırsa 0.0 döner.

        Returns:
            MSE loss değeri (float). Monitoring için kullanılır.
        """
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Gerçekleşen aksiyonlara karşılık gelen Q değerleri
        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN: action selection via q_net, evaluation via target_net.
        # Reduces Q-value overestimation that causes loss spikes.
        with torch.no_grad():
            next_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
            max_next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            targets = rewards_t + self.gamma * max_next_q * (1.0 - dones_t)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — loss spike'larını önler (Q-value overestimation)
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target()

        self.last_loss = loss.item()
        return self.last_loss

    def update_target(self):
        """Target network ağırlıklarını Q-network'ten kopyalar (hard update)."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str):
        """Model checkpoint kaydeder (ağırlıklar + optimizer + epsilon + adım sayısı).

        Args:
            path: Kaydedilecek .pth dosya yolu.
        """
        torch.save({
            "q_net":      self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "step_count": self.step_count,
            "epsilon":    self.epsilon,
        }, path)

    def load(self, path: str):
        """Checkpoint yükler. Eğitime kaldığı yerden devam etmek veya evaluation için.

        Args:
            path: Yüklenecek .pth dosya yolu.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step_count = checkpoint["step_count"]
        self.epsilon    = checkpoint["epsilon"]
