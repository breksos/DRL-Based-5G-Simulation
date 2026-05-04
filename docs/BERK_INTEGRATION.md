# Berk İçin Entegrasyon & Durum Takibi

Yaşayan döküman. Her fix turundan / eğitimden / review'dan sonra güncelle.

**Son güncelleme:** 2026-05-04
**Branch:** main
**Reviewer:** PM review tur 2 tamamlandı. Yapılan iş listeleri için bkz. [FIX_TASKS.md](FIX_TASKS.md) ve [FOLLOWUP_TASKS.md](FOLLOWUP_TASKS.md).

---

## Durum Özeti

| Alan | Durum |
|---|---|
| Channel model + env | ✅ Stabil, testli |
| Baseline'lar (Random / RR / PF) | ✅ Tur 1'de bug'lar düzeltildi |
| DQN training pipeline | ⚠️ Stabil (Double-DQN + Huber), ama A az eğitilmiş (500 ep), B/C/D eski reward ölçeğinde |
| **DQN action adapter** | ❌ **Per-RB API var ama runtime'da hâlâ tek-UE seçiyor** — açık sorunlara bak |
| GUI | ✅ İstenmeyen refactor sonrası eski haline döndürüldü |
| Testler (`pytest tests/`) | ✅ 6 test geçiyor |
| Karşılaştırma artefaktı | ✅ `results/baselines/comparison.csv` var, ama her senaryoda DQN < PF |
| Docs / README | ⚠️ summary.md hâlâ per-RB allocation iddia ediyor — runtime'da doğru değil |

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

---

## Açık sorunlar (öncelik sırasına göre)

### 1. ⛔ ActionAdapter runtime'da hâlâ tek-UE allocation üretiyor
`LAMBDA = 0.5` ve `0.01` magnitude'lu tiebreak gürültüsü, gerçek Q-değer farklarının (~0.1+) üstesinden gelmek için çok küçük. Sonuç: dominant UE her RB'yi alıyor → her senaryoda fairness ≈ `1/n_ue`. PM sanity check'i (fairness > `1/n_ue + 0.05`) **dört senaryoda da fail oluyor**.

**Çözüm seçenekleri:**
- `LAMBDA`'yı yükselt (Q-değerleri [0,1]'de ölçeklendiyse 0.1) ve gürültü magnitude'ını yükselt (örn. 0.5).
- Ya da argmax yerine **softmax over scores → her RB için multinomial sample**.
- `tests/test_adapter.py::test_action_adapter_per_rb`'yi güncelle: çıktıda en az 2 farklı UE indeksi olduğunu assert et (mevcut test `q_values[3] = 0.009` koyarak yapay olarak gürültü aralığına oturtuyor).

### 2. ⛔ Eski checkpoint'ler — B/C/D hâlâ eski reward ölçeğinde
B/C/D `model_final.pth` dosyaları env normalize edilmeden önce eğitilmişti. İçlerindeki Q-değerleri mevcut training ile ~50× farklı ölçekte. Load-time uyarısı eklendi ama rapor için dürüst sonuç istiyorsan yeniden eğit:
```
python main.py --scenario b --episodes 2000
python main.py --scenario c --episodes 2000
python main.py --scenario d --episodes 2000
```
Her log'da max loss < 100 olduğunu doğrula.

### 3. ⚠️ Senaryo A az eğitilmiş (500 ep, orijinal 3676'ya karşı)
Tur 2 retrain'inden sonra A'nın `model_final.pth`'i yetersiz eğitildi. `comparison.csv`'de throughput 60.4 → 53.0 Mbps düştü. Sorun #1 düzeldikten sonra A'yı B/C/D ile aynı bütçede (~2000 ep) yeniden eğit.

### 4. ⚠️ Dökümantasyon DQN davranışını abartıyor
[summary.md](summary.md) action'ı *"50 integer'lık array, her RB için bir tane"* olarak tanımlıyor. Şu an DQN aslında tek bir int üretiyor (dominant UE), o da tüm RB'lere replicate ediliyor. Ya önce sorun #1'i düzelt ya da ilgili bölümleri rapor dürüst olacak şekilde yeniden yaz.

### 5. ⚠️ `gui/app.py` revert'i staged ama commit'lenmemiş
```
git restore --staged gui/app.py
```
Sonra hazır olduğunda bilinçli olarak commit'le.

### 6. 📝 Minor — adapter performansı
`np.random.default_rng(rb_idx)` `agents/train.py`'da per-RB loop içinde kuruluyor. `__init__`'te bir kere noise matrisini precompute et veya loop dışına çıkar.

---

## Karar bekleyen

- **"Tek-UE DQN"'i kabul edip dökümanları mı yeniden yazıyoruz**, yoksa **adapter'ı düzeltip her şeyi yeniden eğitiyor muyuz?** Düzeltme yolu ~1 saat ek dev + ~30 dk eğitim. Döküman yeniden yazma yolu ~15 dk ama projenin RL değer önerisini öldürür.

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
- `results/logs/train_scenario_a.csv` — güncel (500 ep, max loss 4.3, Huber + Double-DQN + LR/2)
- `results/baselines/comparison_OLD.csv` — adapter-rewrite öncesi sayılar
- `results/baselines/comparison.csv` — güncel sayılar (sorun #1 yüzünden hâlâ ≈ PF davranışı)
