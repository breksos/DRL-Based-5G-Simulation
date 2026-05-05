# Developer Agent Task List — Round 4

You are a developer agent (Sonnet 4.6, medium effort) at the repo root `C:\Users\ASUS\Desktop\Projeler\5G Simulation`. Round 3 left DQN beating PF on combined reward but losing to Round Robin in all 4 scenarios. This round closes that gap. Read the whole file first, then execute in order.

**Background.** Combined R = `0.5·min(tp/50, 1.0) + 0.5·fairness`. Current scores from [results/baselines/comparison.csv](../results/baselines/comparison.csv):

| Scenario | RR | DQN | Gap |
|---|---|---|---|
| a | 0.591 | 0.562 | -0.029 |
| b | 0.731 | 0.705 | -0.026 |
| c | 0.807 | 0.729 | -0.078 |
| d | 0.577 | 0.500 | -0.077 |

DQN's fairness (0.40–0.52) is too low vs RR (0.60–0.64). Diagnosis: in [agents/train.py:23](../agents/train.py:23), `LAMBDA = 0.5` makes the SNR-channel-quality term dominate the learned Q-values inside the adapter. The agent's policy never gets to drive allocation toward fairness because SNR keeps pulling it toward channel-greedy. Also, `epsilon_decay_steps=50000` (~250 episodes) means most of 2000 episodes runs at ε=0.01 — too little exploration with a poor policy.

No new dependencies. No commits. Do not modify `gui/app.py`. Comments terse English.

---

## Task 1 — Lower LAMBDA to 0.1

**File:** [agents/train.py:23](../agents/train.py)

Change `LAMBDA = 0.5` → `LAMBDA = 0.1`.

This shrinks the SNR term in `scores = q_values + LAMBDA * snr_norm` so the learned Q-policy dominates the multinomial distribution. The agent can now actually shape per-RB allocation toward whatever the reward signal favours (including fairness).

**Verify.** [tests/test_adapter.py](../tests/test_adapter.py) `test_action_adapter_snr_aware` may need to be checked — with LAMBDA=0.1 and uniform Q-values + one high-SNR UE, the high-SNR UE should still get more RBs than uniform random (~5/50). The test asserts `> 25` which may now fail. If it does, lower the threshold to `> 8` (still well above the uniform-random expectation of 5). Do not lower it further than 8.

---

## Task 2 — Bump exploration

**Files:** [configs/scenario_a.yaml](../configs/scenario_a.yaml), [configs/scenario_b.yaml](../configs/scenario_b.yaml), [configs/scenario_c.yaml](../configs/scenario_c.yaml), [configs/scenario_d.yaml](../configs/scenario_d.yaml)

Change `epsilon_decay_steps: 50000` → `epsilon_decay_steps: 200000` in **all four** scenario configs. With max_steps=200 × 2000 episodes = 400k total steps, a 200k decay schedule means epsilon decays linearly across the first half of training and floors for the second half — a much healthier exploration profile than the current "decay finishes in episode 250" behaviour.

Leave `n_episodes` at 2000 in each config (don't extend training time, just shift exploration).

---

## Task 3 — Retrain all 4 scenarios IN PARALLEL

This is critical for wall-time. Sequential = ~40 minutes; parallel = ~10–15 minutes.

### Pre-flight (do these first, sequentially)

For each scenario in `[a, b, c, d]`:
1. Move existing log: `results/logs/train_scenario_{X}.csv` → `results/logs/train_scenario_{X}_R4OLD.csv` (do NOT overwrite earlier `_R3OLD`, `_OLD`, `_OLD2`, `_LRhalf*` files).
2. Move existing checkpoint: `results/models/scenario_{X}/model_final.pth` → `results/models/scenario_{X}/model_final_R4OLD.pth`.

### Parallel training

Launch all 4 training runs in background simultaneously, then wait. Use the `Bash` tool's `run_in_background=true` parameter for each:

```
Bash run_in_background=true: python main.py --scenario a --episodes 2000
Bash run_in_background=true: python main.py --scenario b --episodes 2000
Bash run_in_background=true: python main.py --scenario c --episodes 2000
Bash run_in_background=true: python main.py --scenario d --episodes 2000
```

All four kicked off in a single message (one parallel batch of 4 Bash calls). After launching, monitor with `BashOutput` checks every ~3 minutes. Do not poll faster than that.

### Acceptance criteria per scenario

After each finishes:
- New `results/logs/train_scenario_{X}.csv` exists with 2000 rows.
- Max loss in column 5 must be < 100. **If max loss >= 100 for any scenario, halve LR in that scenario's config and retrain only that one.** Don't iterate more than once per scenario.

### Resource note

If your machine has fewer than 4 logical cores, this will thrash and be slower than sequential. If `nproc` (or equivalent on Windows: `echo %NUMBER_OF_PROCESSORS%` in cmd, or `[Environment]::ProcessorCount` in PowerShell) reports < 4, fall back to running 2 at a time (a+b in parallel, then c+d in parallel). Document which mode you used in the final report.

---

## Task 4 — Re-run baselines and verify

1. Move `results/baselines/comparison.csv` → `results/baselines/comparison_R3OLD.csv`.
2. Run `python scripts/run_baselines.py`.
3. Compute combined R per row using `R = 0.5·min(tp/50, 1.0) + 0.5·fairness`.

### Required outcome

DQN must beat all three baselines on combined R in **at least 3 of 4 scenarios**.

### If sanity check fails

Do NOT iterate further. Stop and report. The user explicitly chose this fix as "fastest at hand" — if it doesn't land, that's a meaningful finding.

---

## Task 5 — Update docs only if Task 4 passed

If DQN now wins ≥3/4 on combined R:
- [docs/summary.md](summary.md) — update the per-scenario table at line ~208 with new numbers.
- [docs/BERK_INTEGRATION.md](BERK_INTEGRATION.md) — change open issue #2 ("DQN combined-R'da RR'ı geçemiyor") status. If RR is now beaten in 3/4 or 4/4, remove the issue. If 3/4, replace with a one-line note on which scenario DQN still loses.
- Add a Tur 4 entry under "Yapılanlar".

If DQN did NOT meet the criteria, leave the docs untouched and report failure.

---

## Out of scope

- Do not modify `gui/app.py`.
- Do not extend training beyond 2000 episodes.
- Do not change reward formulation, network architecture, or buffer size.
- Do not add new tests beyond the threshold tweak in Task 1.
- Do not commit, push, or open PRs.

---

## Final report

Print to stdout, under 500 words:

1. **Tasks completed** by ID, with one-line note each. Note whether you used full 4-way parallel or 2+2 fallback in Task 3.
2. **Stability table:** scenario, max loss, last-100 avg reward.
3. **Combined-R table:** scenario × scheduler grid with combined R values. Star the winner per scenario.
4. **Verdict:** Did DQN beat all 3 baselines on combined R in ≥3/4 scenarios? If no, in how many?
5. `pytest tests/` output (one line).
6. List of files modified.
7. Anything surprising you noticed during training (loss spikes that recovered, reward plateaus, etc.) — keep brief.
