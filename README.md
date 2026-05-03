# Deep Reinforcement Learning Based Resource Allocation in 5G

**CSE 476 · Mobile Communication Networks**  
İsmail Kabak · Berk Hakan Öge · Emre Can Tuncer

---

## Quick Start

```bash
pip install -r requirements.txt

# Launch GUI
python main.py --gui

# 10-step terminal demo
python main.py --demo

# Train DQN on a scenario
python main.py --scenario a

# Run stability test
python scripts/stability_test.py

# Run robustness / edge-case analysis
python scripts/robustness_test.py
```

---

## Project Structure

```
project/
├── env/
│   ├── channel_model.py     # Path loss, SNR, Shannon capacity
│   ├── network_env.py       # Gymnasium FiveGEnv (state/action/reward)
│   └── test_channel_model.py
├── agents/                  # DQN agent (İsmail)
├── baselines/
│   └── schedulers.py        # Random, Round Robin, Proportional Fair
├── evaluation/
│   └── metrics.py           # Jain's fairness, throughput helpers
├── gui/
│   └── app.py               # Real-time simulation GUI
├── scripts/
│   ├── stability_test.py    # Phase 3: 1000-episode NaN/inf check
│   └── robustness_test.py   # Phase 4: edge cases, cross-scenario
├── configs/
│   ├── default.yaml
│   ├── scenario_a.yaml      # 10 UE / 50 RB
│   ├── scenario_b.yaml      # 15 UE / 75 RB
│   ├── scenario_c.yaml      # 20 UE / 100 RB
│   └── scenario_d.yaml      # 20 UE / 50 RB (resource-scarce)
├── results/                 # Training logs, model checkpoints, plots
├── utils/
│   └── config.py            # YAML config loader
└── main.py
```

---

## Environment Interface

```python
from env.network_env import FiveGEnv

env = FiveGEnv(n_ue=10, n_rb=50, cell_radius=500.0)
state, _   = env.reset()                # state shape: (2*N_UE + N_RB,)
action     = env.action_space.sample()  # shape: (N_RB,), values 0..N_UE-1
state, reward, terminated, truncated, info = env.step(action)
# info keys: throughput_total (bps), fairness (Jain's index)
```

**State vector:** `[snr_1..snr_N | throughput_1..throughput_N (Mbps) | rb_occupancy_1..rb_N_RB]`

**Reward:** `α · Σ throughput_Mbps  +  β · Jain's Index`  (default α=β=0.5)

---

## Channel Model

Log-distance path loss calibrated for 50–500 m micro-cell deployment:

```
PL(d) = 70.25 + 37.6 · log₁₀(d)        [d in metres]
SNR   = P_tx − PL(d) − N₀              [dB, clamped ≥ 0]
C     = B · log₂(1 + SNR_linear)        [bps per RB]
```

| Parameter | Value |
|---|---|
| P_tx | 46 dBm |
| Path loss exponent n | 3.76 |
| RB bandwidth | 180 kHz (12 × 15 kHz) |
| UE distance range | 50–500 m |
| SNR @ 100 m | ≈ 22 dB |
| SNR @ 500 m | ≈ 0 dB |

---

## Scenarios

| ID | N_UE | N_RB | Description |
|---|---|---|---|
| A | 10 | 50 | Baseline |
| B | 15 | 75 | Medium load |
| C | 20 | 100 | Dense |
| D | 20 | 50 | Resource-scarce |

---

## GUI Controls

| Control | Description |
|---|---|
| Scenario | Switch between A/B/C/D |
| Scheduler | Random / Round Robin / Proportional Fair |
| Mobility | Enable ±10 m random walk per episode |
| Step | Advance one step |
| Run / Pause | Start/stop auto-run |
| Speed slider | 20–800 ms per step |
