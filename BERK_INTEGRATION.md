# Berk İçin Entegrasyon & Doğrulama Rehberi

## ÖNEMLİ — Claude Code İçin

**Bu projeye büyük değişiklik yapma.** Aşağıdaki doğrulama adımlarını takip et.
Çalışan kod var, bozma. Sadece listelenen sorunları düzelt.

---

## Proje Durumu

İsmail'in kodu (`agents/`) tamamlandı ve GitHub'da. 4 senaryo için eğitilmiş modeller
`results/models/` altında mevcut. GUI çalışıyor (`python main.py --gui`).

### Klasör Sorumlulukları
| Klasör | Sahibi | Durum |
|--------|--------|-------|
| `agents/` | İsmail | Tamamlandı, dokunma |
| `env/` | Berk | Aktif — doğrulama gerekli |
| `baselines/` | Emre | Aktif |
| `evaluation/` | Emre | Aktif |
| `gui/` | Berk | Aktif |
| `configs/` | Ortak | Tamamlandı |
| `utils/` | Berk | Tamamlandı |

---

## Doğrulanması Gereken Entegrasyon Noktaları

### 1. Demo çalışıyor mu?
```bash
python main.py --demo
```
Beklenen: 10 satır `Step / Reward / Throughput / Fairness` çıktısı.

### 2. Senaryo A eğitimi çalışıyor mu?
```bash
python main.py --scenario a --episodes 50
```
Beklenen: 50 episode tamamlanır, `results/models/scenario_a/` altına checkpoint kaydedilir.

### 3. GUI açılıyor mu?
```bash
python main.py --gui
```
Beklenen: Pencere açılır. Scheduler dropdown'da **DQN Agent** seçeneği var.
DQN Agent seçilince başlık `DQN (model_final.pth)` veya `DQN (ep XXXX)` olmalı.

### 4. DQN Agent GUI'de çalışıyor mu?
GUI'de: Scenario A → Scheduler: DQN Agent → ▶ Run
Beklenen: RB Allocation heatmap güncelleniyor, Throughput grafikleri değişiyor.

### 5. Interface sözleşmesi doğru mu?
```python
from env.network_env import FiveGEnv
env = FiveGEnv(n_ue=10, n_rb=50)
state, _ = env.reset()
assert state.shape == (70,)          # 2*10 + 50

action = env.action_space.sample()
assert action.shape == (50,)         # N_RB boyutlu

s, r, term, trunc, info = env.step(action)
assert isinstance(r, float)
assert 'throughput_total' in info
assert 'fairness' in info
```

---

## Bilinen Durumlar (Hata Değil)

- `results/models/scenario_a/model_ep*.pth` dosyaları çok → `.gitignore`'da ignore ediliyor, sadece `model_final.pth` commit'leniyor. Normal.
- Loss değerleri bazen büyük çıkıyor (10-100 arası) → reward normalizasyonu uygulandı, kabul edilebilir.
- `agents/__pycache__/` klasörü → ignore ediliyor, önemli değil.

---

## Kesinlikle Dokunulmaması Gerekenler

- `agents/dqn_agent.py` — DQN mimarisi tamamlandı
- `agents/train.py` — eğitim döngüsü çalışıyor
- `agents/dqn_scheduler.py` — GUI entegrasyonu çalışıyor
- `configs/scenario_*.yaml` — hiperparametreler kararlaştırıldı
- `results/models/*/model_final.pth` — eğitilmiş modeller, silme

---

## Sorun Çıkarsa

`python main.py --gui` çalışmıyorsa önce dene:
```bash
pip install -r requirements.txt
python main.py --demo   # env test
```

`DQN Agent` modeli bulunamıyor mesajı çıkıyorsa:
```bash
ls results/models/scenario_a/   # model_final.pth var mı?
```
