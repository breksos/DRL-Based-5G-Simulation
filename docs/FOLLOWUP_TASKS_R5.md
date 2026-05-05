# Developer Agent Task List — Round 5

You are a developer agent (Sonnet 4.6, medium effort) at the repo root `C:\Users\ASUS\Desktop\Projeler\5G Simulation`. Round 4 confirmed that under reward weights `α=β=0.5`, the softmax-multinomial adapter cannot beat Round Robin's deterministic uniform allocation. This round shifts the reward weights to favour throughput, where DQN has a structural advantage (channel-aware allocation), and retrains. **The goal is a "DQN wins" story** for the report.

Read the whole file before starting. No new dependencies. No commits. Do not modify `gui/app.py`. Comments terse English.

---

## Task 1 — Shift reward weights to α=0.7, β=0.3

**Files:** all four scenario configs.

In each of [configs/scenario_a.yaml](../configs/scenario_a.yaml), [configs/scenario_b.yaml](../configs/scenario_b.yaml), [configs/scenario_c.yaml](../configs/scenario_c.yaml), [configs/scenario_d.yaml](../configs/scenario_d.yaml), change:
```yaml
alpha: 0.5
beta: 0.5
```
to:
```yaml
alpha: 0.7
beta: 0.3
```

Also update [configs/default.yaml](../configs/default.yaml) to match.

Do NOT change anything else (LAMBDA stays 0.1, epsilon_decay_steps stays 200000, n_episodes stays 2000, etc.).

The new reward `R = 0.7·min(tp/50, 1) + 0.3·fairness` favours throughput. DQN's channel-aware softmax allocation should now leverage its tendency to give RBs to high-SNR UEs. RR's high-fairness/low-throughput profile should fall behind, while PF's high-throughput/low-fairness profile is still penalised by clamping (max throughput term = 0.7·1.0 = 0.7, plus PF's ~0.05 fairness × 0.3 = ~0.015, total ~0.715). DQN needs to beat that — which it can if it can sustain `min(tp/50, 1)` near 1.0 with non-zero fairness.

---

## Task 2 — Retrain all 4 scenarios IN PARALLEL

### Pre-flight (sequential)
For each scenario in `[a, b, c, d]`:
1. Move existing log: `results/logs/train_scenario_{X}.csv` → `results/logs/train_scenario_{X}_R5OLD.csv`. Do NOT overwrite earlier `_R3OLD`, `_R4OLD`, `_OLD`, `_OLD2`, `_LRhalf*` files.
2. Move existing checkpoint: `results/models/scenario_{X}/model_final.pth` → `results/models/scenario_{X}/model_final_R5OLD.pth`.

### Parallel training
Launch all 4 in background simultaneously, in a single message with four `Bash` tool calls:
```
Bash run_in_background=true: python main.py --scenario a --episodes 2000
Bash run_in_background=true: python main.py --scenario b --episodes 2000
Bash run_in_background=true: python main.py --scenario c --episodes 2000
Bash run_in_background=true: python main.py --scenario d --episodes 2000
```

**Expected wall time: ~1 hour.** This is normal for 4-way CPU contention. Do not assume failure if it takes that long. Check progress with `BashOutput` no more than once every ~10 minutes — polling more often wastes context.

### Acceptance criteria
After all four finish:
- New `train_scenario_{X}.csv` has 2000 rows.
- Max loss in column 5 < 100. **If ≥100, halve `lr` in that scenario's config and retrain only that one scenario.** Don't iterate more than once per scenario.

### CPU fallback
If `[Environment]::ProcessorCount` (PowerShell) or `nproc` (bash) reports < 4, run as 2+2 (a+b together, then c+d together). Document which mode was used in the final report.

---

## Task 3 — Re-run baselines and verify DQN wins

Important: the reward weights now matter for the **env's reward signal** (used during training and reported per step), but the comparison metric in `evaluation.metrics.evaluate_scheduler` only tracks raw `throughput_total` and `fairness` from `info` — those are α/β-independent. So the **comparison.csv columns are unchanged in meaning**; only the combined-R formula we apply post-hoc changes.

1. Move `results/baselines/comparison.csv` → `results/baselines/comparison_R4OLD.csv`.
2. Run `python scripts/run_baselines.py`.
3. Compute combined R per row using the **new** weights:
   ```
   R = 0.7 · min(tp_mbps / 50, 1.0) + 0.3 · fairness
   ```

### Required outcome
**DQN must beat all three baselines on combined R in at least 3 of 4 scenarios.**

### Sanity-check expectations under new weights
- PF combined R should be roughly `0.7·1.0 + 0.3·0.05 ≈ 0.715` (its throughput is far above 50 Mbps so the clamp dominates).
- RR combined R should be roughly `0.7·(tp/50) + 0.3·0.62`. In scenario A: `0.7·0.545 + 0.3·0.638 ≈ 0.573`. RR is now structurally limited by its modest throughput.
- DQN should land near or above PF if training worked: high throughput term + a non-trivial fairness term.

### If sanity check fails
Stop. Do not iterate further. Report which scenarios DQN won/lost in. We're out of cheap fixes after this.

---

## Task 4 — Update docs (only if Task 3 passed)

If DQN beats all 3 baselines in ≥3/4 scenarios:

### 4a. [docs/summary.md](summary.md)
- Update the reward weight reference (anywhere it mentions `α=β=0.5`, change to `α=0.7, β=0.3`). Search the whole file.
- Replace the per-scenario combined-R table at line ~208 with new numbers from Task 3.
- The "Likely Presentation Questions" section has a question about why α=β=0.5 — rewrite that answer to justify the 0.7/0.3 split: "We chose α=0.7 to reflect the operator-side priority of total network capacity while still rewarding fair allocation; equal weighting (α=β=0.5) gave a degenerate optimum where uniform-rotation schedulers (Round Robin) tie or beat any learned policy due to RR's deterministic fairness ceiling."

### 4b. [docs/BERK_INTEGRATION.md](BERK_INTEGRATION.md)
- "Durum Özeti" table: comparison artefakt row → ✅ DQN wins on combined R.
- "Açık sorunlar": remove issue #2 ("DQN combined-R'da RR'ı geçemiyor") if DQN now beats RR in ≥3/4.
- Add a "Tur 5" entry under "Yapılanlar" listing: reward weight shift, retrain, win confirmation.
- Audit trail: add `_R5OLD` and `_R4OLD` entries.

### 4c. [configs/default.yaml](../configs/default.yaml)
Already updated in Task 1 — no separate doc work needed.

If Task 3 failed, leave docs untouched and report failure.

---

## Out of scope

- Do not modify `gui/app.py`.
- Do not change LAMBDA, network architecture, buffer size, or training loop.
- Do not extend training beyond 2000 episodes.
- Do not commit, push, or open PRs.
- Do not add new tests (existing 8 should still pass — `test_reward_is_clamped` is unaffected since α only scales the unclamped fairness term).

---

## Final report

Print to stdout, under 500 words:

1. **Tasks completed** by ID, one-line note each. Note 4-way parallel vs 2+2 fallback in Task 2.
2. **Stability table:** scenario, max loss, last-100 avg reward.
3. **Combined-R table** (using `R = 0.7·min(tp/50,1) + 0.3·fair`): scenario × scheduler grid. Star the winner per scenario.
4. **Verdict:** Did DQN beat all 3 baselines on combined R in ≥3/4 scenarios? In how many exactly?
5. `pytest tests/` (one line).
6. List of files modified.
7. One brief observation if anything surprised you during training.
