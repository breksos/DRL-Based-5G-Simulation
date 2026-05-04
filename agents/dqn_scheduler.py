"""DQNAgent'ı baseline scheduler arayüzüne (allocate/reset) saran wrapper.

GUI ve evaluation kodunun DQN'i diğer scheduler'larla aynı şekilde kullanmasını sağlar.
"""

import os
import glob
import numpy as np

try:
    from agents.dqn_agent import DQNAgent
    from agents.train import ActionAdapter
except ImportError:
    from dqn_agent import DQNAgent
    from train import ActionAdapter


class DQNScheduler:
    """Eğitilmiş DQN modelini scheduler arayüzünde çalıştırır (greedy, ε=0).

    Args:
        n_ue: Kullanıcı sayısı.
        n_rb: Resource Block sayısı.
        model_path: .pth checkpoint dosyası. None ise otomatik bulur.
        scenario: 'a', 'b', 'c', 'd' — model_path None ise bu senaryonun
                  son checkpoint'i yüklenir.
    """

    def __init__(self, n_ue: int, n_rb: int,
                 model_path: str = None, scenario: str = "a"):
        self.n_ue = n_ue
        self.n_rb = n_rb
        self._adapter = ActionAdapter(n_ue=n_ue, n_rb=n_rb)
        state_dim = 2 * n_ue + n_rb

        self._agent = DQNAgent(state_dim=state_dim, action_dim=self._adapter.action_dim)

        path = model_path or self._find_latest(scenario)
        if path and os.path.exists(path):
            self._agent.load(path)
            self._agent.epsilon = 0.0  # greedy — tam öğrenilmiş policy
            self._loaded = True
            self._model_path = path
        else:
            self._loaded = False
            self._model_path = None

    def _find_latest(self, scenario: str) -> str | None:
        """model_final.pth varsa onu, yoksa en son checkpoint'i döndürür."""
        base = os.path.join("results", "models", f"scenario_{scenario}")
        final = os.path.join(base, "model_final.pth")
        if os.path.exists(final):
            return final
        files = glob.glob(os.path.join(base, "model_ep*.pth"))
        if not files:
            return None
        files.sort(key=lambda p: int(os.path.basename(p).replace("model_ep", "").replace(".pth", "")))
        return files[-1]

    def allocate(self, state: np.ndarray) -> np.ndarray:
        """Return N_RB allocation using per-RB scoring (q_values + SNR)."""
        if not self._loaded:
            return np.array([i % self.n_ue for i in range(self.n_rb)], dtype=int)
        q_vals = self._agent.q_values(state)
        return self._adapter.to_allocation(state[:self.n_ue], q_vals)

    def reset(self):
        pass  # DQN stateless — episode başında sıfırlanacak bir şey yok

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_info(self) -> str:
        if self._loaded:
            ep = os.path.basename(self._model_path).replace("model_ep", "").replace(".pth", "")
            return f"ep {ep}"
        return "model bulunamadı"
