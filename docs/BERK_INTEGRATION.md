# Berk İçin Entegrasyon & Durum Takibi

Yaşayan döküman. Her fix turundan / eğitimden / review'dan sonra güncelle.

**Son güncelleme:** 2026-05-05
**Branch:** main
**Reviewer:** PM review tur 3 tamamlandı. Yapılan iş listeleri için bkz. [FIX_TASKS.md](FIX_TASKS.md), [FOLLOWUP_TASKS.md](FOLLOWUP_TASKS.md) ve [FOLLOWUP_TASKS_R3.md](FOLLOWUP_TASKS_R3.md).

---

## Durum Özeti

| Alan | Durum |
|---|---|
| Channel model + env | ✅ Stabil, testli |
| Baseline'lar (Random / RR / PF) | ✅ Tur 1'de bug'lar düzeltildi |
| DQN training pipeline | ✅ Tüm senaryolar (A/B/C/D) yeniden eğitildi, max loss < 100 |
| **DQN action adapter** | ✅ Softmax+multinomial per-RB allocation, tekil RNG (seed destekli) |
| GUI | ✅ İstenmeyen refactor sonrası eski haline döndürüldü |
| Testler (`pytest tests/`) | ✅ 8 test geçiyor |
| Karşılaştırma artefaktı | ✅ `results/baselines/comparison.csv` güncel (R3 modelleri), DQN PF'i geçiyor |
| Docs / README | ✅ Güncellendi (summary.md gerçek sayılarla) |

---

## Yapılanlar

### Tur 1 ([FIX_TASKS.md](FIX_TASKS.md))
- [x] Round-Robin scheduler artık dönüyor (`+= 1`)
- [x] Reward normalizasyonu birleştirildi — env içinde tek `THROUGHPUT_NORM_MBPS = 50`
- [x] PF lazy-init — ilk çağrıda UE-0 bias'ı yok
- [x] Mobility dökümanı koddaki davranışla eşleşecek şekilde güncellendi
- [x] Double DQN target hesaplama
- [x] `target_update_freq` 1000 → 500'e düşürüldü (tüm config'lerde)
- [x] `RandomScheduler` artık `seed` kabul ediyor
- [x] `MockEnv` `agents/_mock_env.py`'a taşındı
- [x] Defansif `% n_ue` yerine assertion'lar kondu
- [x] `tests/` klasörü oluşturuldu — `test_baselines.py`, `test_adapter.py`
- [x] `scripts/run_baselines.py` yazıldı — `results/baselines/comparison.csv` üretir
- [x] `results/models/TRAINING_NOTES.md` gerçek episode sayılarını dökümanlıyor

### Tur 2 ([FOLLOWUP_TASKS.md](FOLLOWUP_TASKS.md))
- [x] `DQNAgent.q_values(state)` metodu
- [x] Yeni `ActionAdapter.to_allocation(snrs, q_values)` imzası
- [x] Adapter çağıranları `dqn_scheduler.py` ve `main.py`'da güncellendi
- [x] Senaryo A 500 ep yeniden eğitildi — max loss **4.3** (Huber loss + LR yarıya: 2.5e-4)
- [x] GUI değişiklikleri geri alındı
- [x] `DQNAgent.load`'a reward-scale uyumsuzluğu uyarısı eklendi
- [x] Yeni `comparison.csv` yanında `comparison_OLD.csv` saklandı

### Tur 3 ([FOLLOWUP_TASKS_R3.md](FOLLOWUP_TASKS_R3.md))
- [x] Reward clamping — `norm_tp = min(tp/50, 1.0)` → adım başı reward artık `[0, 1]` aralığında
- [x] Per-RB RNG bug düzeltildi — `ActionAdapter` artık tek `self._rng` kullanıyor (seed destekli)
- [x] `DQNScheduler` ve `scripts/run_baselines.py` seed=0 ile güncellendi
- [x] Tüm senaryolar 2000 ep yeniden eğitildi (B/C/D LR yarıya: 2.5e-4; C ikinci kez: 1.25e-4)
- [x] Yeni `test_reward_is_clamped` ve `test_action_adapter_distribution` testleri eklendi (8 test, hepsi geçiyor)
- [x] `comparison_R2OLD.csv` saklandı, yeni `comparison.csv` oluşturuldu

---

## Açık sorunlar (öncelik sırasına göre)

### 1. ⚠️ `gui/app.py` revert'i staged ama commit'lenmemiş
```
git restore --staged gui/app.py
```
Sonra hazır olduğunda bilinçli olarak commit'le.

### 2. ⚠️ DQN combined-R'da RR'ı geçemiyor
R3 sonrası DQN, PF'i tüm senaryolarda geçiyor (reward clamping sayesinde) ama Round Robin'i geçemiyor. RR'ın üstünlüğü yüksek fairness'tan geliyor (0.60–0.64 vs DQN 0.40–0.52). Daha uzun eğitim veya farklı reward şekillendirmesi gerekebilir.

---

---

## Manuel doğrulama listesi (Berk)

Teslimden önce lokalde çalıştır:

- [ ] `python main.py --demo` — 10 adımlık run reward / throughput / fairness yazıyor
- [ ] `python -m pytest tests/` — hepsi yeşil
- [ ] `python scripts/run_baselines.py` — tamamlanır, 16 satırlık CSV yazar
- [ ] `python main.py --gui` — pencere açılır
  - [ ] Dört scheduler de seçilebiliyor (Random / RR / PF / DQN Agent)
  - [ ] **DQN heatmap çok-renkli kümeler gösteriyor** (sorun #1 düzeldi demektir)
  - [ ] Mobility toggle reset'ler arasında UE pozisyonlarını yumuşakça değiştiriyor
  - [ ] Senaryo değişiminde crash yok
- [ ] Interface contract hâlâ geçerli:
  ```python
  from env.network_env import FiveGEnv
  env = FiveGEnv(n_ue=10, n_rb=50)
  state, _ = env.reset()
  assert state.shape == (70,)
  action = env.action_space.sample()
  s, r, term, trunc, info = env.step(action)
  assert {"throughput_total", "fairness"} <= info.keys()
  ```

---

## Klasör Sorumlulukları

| Klasör | Sahibi | Notlar |
|---|---|---|
| `agents/` | İsmail (orijinal) | PM/dev turlarında değiştirildi — Double-DQN, Huber, q_values metodu, adapter rewrite |
| `env/` | Berk | Tur 1'de reward normalizasyonu eklendi |
| `baselines/` | Emre | RR + PF + Random bug'ları tur 1'de düzeltildi |
| `evaluation/` | Emre | Defansif modulo assert'lere çevrildi |
| `gui/` | Berk | Tur 2'de revert edildi — sorun #1 düzelene kadar tekrar refactor etme |
| `configs/` | Ortak | `target_update_freq=500`, senaryo A LR yarıya |
| `scripts/` | Berk | `run_baselines.py` eklendi |
| `tests/` | Yeni | Tur 1'de eklendi |
| `utils/` | Berk | Dokunulmadı |

---

## Audit trail

- `results/logs/train_scenario_a_OLD.csv` — orijinal 3676 ep eğitim (max loss 512k, MSE)
- `results/logs/train_scenario_a_OLD2.csv` — başarısız ilk retry (max loss 131.8, MSE + Double-DQN)
- `results/logs/train_scenario_a_R3OLD.csv` — R3 öncesi (500 ep, max loss 4.3, Huber + LR/2)
- `results/logs/train_scenario_b_R3OLD.csv` — R3 öncesi B (eski reward ölçeği)
- `results/logs/train_scenario_b_LRhalf1.csv` — R3 ilk deneme B (max loss 325, lr=5e-4)
- `results/logs/train_scenario_c_R3OLD.csv` — R3 öncesi C (eski reward ölçeği)
- `results/logs/train_scenario_c_LRhalf1.csv` — R3 ilk deneme C (max loss 1596, lr=5e-4)
- `results/logs/train_scenario_c_LRhalf2.csv` — R3 ikinci deneme C (max loss 250, lr=2.5e-4)
- `results/logs/train_scenario_d_R3OLD.csv` — R3 öncesi D (eski reward ölçeği)
- `results/logs/train_scenario_d_LRhalf1.csv` — R3 ilk deneme D (max loss 749, lr=5e-4)
- `results/baselines/comparison_OLD.csv` — adapter-rewrite öncesi sayılar
- `results/baselines/comparison_R2OLD.csv` — R3 öncesi sayılar (clampsiz reward)
- `results/baselines/comparison.csv` — güncel R3 sayıları (clamped reward, softmax adapter)
