# 5G DRL Project — Berk's Build Plan

**Project:** Deep Reinforcement Learning Based Resource Allocation in a Simplified 5G Downlink Network  
**Berk's domain:** `env/` — 5G environment simulation + GUI

---

## Folder Structure ✓

```
project/
├── env/              ← Berk's area
├── agents/           ← İsmail's area
├── baselines/        ← Emre's area (stubs provided for GUI)
├── evaluation/       ← Emre's area (stubs provided for GUI)
├── gui/              ← Berk's area
├── scripts/          ← Berk's area
├── utils/
├── results/
├── configs/
├── main.py
└── requirements.txt
```

---

## Phase 1 — Channel Model ✓

**Output:** `env/channel_model.py`

- [x] Setup: `requirements.txt`, folder structure
- [x] `path_loss_db(d)` — log-distance model (calibrated PL0=70.25 for 50–500m range)
- [x] `snr_db(d)` — P_tx=46 dBm, thermal noise per RB, clamp at 0
- [x] `capacity_bps(snr_db, B)` — Shannon formula
- [x] 6/6 unit tests pass: 100m→22 dB, 500m→0 dB, monotone, no negatives, array ops

**Interface contract (agreed with team):**
```python
env.reset() → state  # np.array, shape: (2*N_UE + N_RB,)
env.step(action) → (next_state, reward, done, truncated, info)
# action: np.array shape (N_RB,), each element in 0..N_UE-1
```

---

## Phase 2 — Gymnasium Environment ✓

**Output:** `env/network_env.py`

- [x] `class FiveGEnv(gymnasium.Env)` with bounded obs space (70-dim for default)
- [x] `reset()` — random UE distances, SNR, zero throughputs, return state
- [x] `step(action)` — per-UE throughput, reward = α·ΣMbps + β·Jain, truncated at max_steps
- [x] `gymnasium.utils.env_checker` passes (no errors)
- [x] 100 random steps: no NaN/inf, reward ~13–19, fairness ~0.3–0.7
- [x] `main.py` entry point + `utils/config.py` YAML loader working
- [x] `distances` property exposed for GUI
- [x] Mobility model: `env.mobility = True` enables ±10m random walk between episodes

---

## Phase 3 — Multi-Scenario Configs ✓

**Output:** `configs/scenario_*.yaml`

- [x] Scenario A: 10 UE, 50 RB
- [x] Scenario B: 15 UE, 75 RB
- [x] Scenario C: 20 UE, 100 RB
- [x] Scenario D: 20 UE, 50 RB
- [x] Stability test: 1000 episodes × 4 scenarios — 0 NaN, 0 Inf — all PASSED
- [x] Mobility model implemented (±10m per episode reset)

---

## Phase 4 — Edge Cases & Robustness ✓

**Output:** `scripts/robustness_test.py`, `configs/edge_*.yaml`

- [x] Edge case A: all UEs close (cell_radius=100m)
- [x] Edge case B: all UEs far (standard radius, high distances)
- [x] Edge case C: mixed (half close 50–150m, half far 400–500m)
- [x] Robustness test: cross-scenario evaluation (10 UE policy on 15/20 UE envs)
- [x] Cell radius sweep: 200m / 500m / 1000m results
- [x] Path-loss exponent sweep: n=2.5 vs n=3.76

---

## Phase 5 — Code Cleanup & Report Section ✓

- [x] Docstrings + type hints in `env/channel_model.py` and `env/network_env.py`
- [x] 3GPP calibration note in channel_model.py docstring
- [x] `README.md` — install, run, structure, env interface, parameter table
- [x] `baselines/schedulers.py` — Random, Round Robin, Proportional Fair (interfaces ready for İsmail's agent)
- [x] `evaluation/metrics.py` — `jains_fairness`, `per_ue_throughput`, `evaluate_scheduler`

---

## Phase 6 — GUI & Final ✓

**Output:** `gui/app.py`, `python main.py --gui`

- [x] Real-time simulation GUI with dark theme
- [x] 5 live plots: Network Topology · SNR per UE · Throughput per UE · RB Allocation heatmap · Live History
- [x] Controls: Scenario selector, Scheduler selector, Mobility toggle, Step/Run/Pause, Speed slider
- [x] Stats panel: Episode, Step, Ep. Reward, Throughput, Fairness
- [x] `python main.py --gui` launches the GUI
- [x] `python main.py --demo` 10-step terminal demo

---

## Key Numbers

| Parameter | Value |
|---|---|
| P_tx | 46 dBm |
| Path loss exponent n | 3.76 |
| UE distance range | 50–500 m |
| Default N_UE / N_RB | 10 / 50 |
| State dim (default) | 2×10 + 50 = 70 |
| Reward weights | α=0.5, β=0.5 |

---

## Integration Points (sync with team)

| End of Phase | Status |
|---|---|
| Phase 1 | ✓ channel_model.py works, interface contract agreed |
| Phase 2 | ✓ FiveGEnv passes env_checker — İsmail/Emre can plug in |
| Phase 3 | ✓ All configs ready, stability verified |
| Phase 4 | ✓ Edge cases + robustness scripts ready |
| Phase 5 | ✓ Clean code, README, metric stubs for Emre |
| Phase 6 | ✓ GUI live, demo working |

## Handoff Notes for Team

- **İsmail**: plug `FiveGEnv` into `agents/train.py`. State dim = `2*N_UE + N_RB`. Load config with `utils.config.load_config`.
- **Emre**: `evaluation/metrics.py` has `jains_fairness`, `evaluate_scheduler` ready. `baselines/schedulers.py` has all three schedulers with `allocate(state) → action` interface.
