# 5G Simulation GUI — User Manual

Launch:

```bash
python main.py --gui
```

The window has a **left control panel** and a **right plot area** (5 live plots).

---

## Left Panel — Controls

| Control | What it does |
|---|---|
| **Scenario** dropdown | A (10 UE / 50 RB), B (15/75), C (20/100), D (20/50, scarce). Switching rebuilds the environment and reloads the matching DQN model. |
| **Scheduler** dropdown | Random, Round Robin, Proportional Fair, **DQN Agent**. DQN auto-loads `results/models/scenario_<x>/model_final.pth`. |
| **Mobility** checkbox | When ON, UE positions drift ±10 m at each episode reset (random walk). |
| **Reset Env** | Resamples UE positions, zeroes throughput history, episode counter +1. |
| **Step** | Advances exactly one time step (1 ms). |
| **▶ Run / ⏸ Pause** | Toggles continuous stepping. |
| **Speed (ms/step)** slider | Delay between auto-steps. Range **20–800 ms**. Lower = faster. |

### Stats readout

- **Episode** — current episode number (resets bump this).
- **Step** — `current / max_steps` (default `… / 200`).
- **Ep. Reward** — accumulated reward this episode.
- **Throughput** — total Mbps delivered this step (sum across all UEs).
- **Fairness** — Jain's index for this step (0–1, higher = more even).

---

## Right Panel — The 5 Plots

### 1. Network Topology
- gNB at center (★), UEs as dots, colored by SNR (red = weak, green = strong).
- **Look for:** green dots clustered near center, red near the cell edge. Mobility ON → dot positions shift slightly between episodes.

### 2. SNR per UE (bar chart)
- One bar per UE, height = current SNR in dB.
- **Comment template:** "UE 3 has 22 dB (close), UE 7 has 0 dB (edge). Wide spread → interesting scheduling problem."

### 3. Throughput per UE (bar chart)
- Mbps each UE received this step.
- **Random / RR:** many bars non-zero, roughly even.
- **Proportional Fair:** typically one tall bar (the winner this step), rest near zero.
- **DQN:** several bars, biased toward high-SNR UEs but not single-winner.

### 4. RB Allocation (heatmap)
- X-axis = RB index, Y-axis = a single row, color = which UE got the RB.
- **Random:** rainbow noise.
- **RR:** repeating cyclic pattern.
- **PF:** one solid color across all RBs (one winner per step).
- **DQN:** multi-color clusters — the agent is grouping RBs to a few favored UEs.
- The colorbar covers all UE indices (every UE for ≤12 UEs, every other index for more).

### 5. Live History (Reward & Fairness over last 200 steps)
- Two line traces. Reward (top) and Fairness (bottom).
- **DQN during training:** rising curve early, plateau later.
- **DQN at inference (what the GUI shows):** roughly stable plateau.
- **PF:** high reward, very low fairness line (~0.1).
- **RR:** moderate reward, fairness around 0.6.

---

## Typical Workflow

1. Launch GUI.
2. Pick **Scenario A**, Scheduler **Round Robin** → Run for ~50 steps. Note baseline numbers.
3. Switch to **Proportional Fair** → notice fairness collapse, throughput similar.
4. Switch to **DQN Agent** → reward should be visibly higher than both baselines.
5. Toggle **Mobility ON**, hit **Reset Env**, watch topology dots shift each episode.
6. Try **Scenario D** (scarce RBs) — the trade-off becomes the sharpest; DQN's advantage should be clearest here on the combined reward.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Title says `DQN Agent (no model found)` | Train it: `python main.py --scenario a --episodes 2000`, or check `results/models/scenario_<x>/model_final.pth` exists. |
| GUI freezes during Run | Speed too low (close to 20 ms) on slow hardware — drag slider right. |
| Colors don't match between plots | All plots use the same `tab20` colormap by UE index — should already be consistent. |
| Window too small | Resize manually; matplotlib axes will scale. |

---

## Validated Test Results (sanity reference)

20 episodes × 200 steps, mean per step:

| Scenario | Random reward | RR reward | PF reward | DQN reward |
|---|---|---|---|---|
| A (10/50) | 14.89 | 14.97 | 14.70 | **24.66** |
| B (15/75) | 23.16 | 23.19 | 23.45 | **31.03** |
| C (20/100) | 29.55 | 29.65 | 29.36 | **30.12** |
| D (20/50) | 14.89 | 14.97 | 14.69 | **19.24** |

DQN beats every baseline on the combined reward across all four scenarios.
