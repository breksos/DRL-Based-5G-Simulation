# Developer Agent Follow-up Task List — Round 2

You are a developer agent (Sonnet 4.6, medium effort) working on the **5G DRL Resource Allocation** project at the repo root `C:\Users\ASUS\Desktop\Projeler\5G Simulation`. This is a follow-up to the previous round of fixes (see [docs/FIX_TASKS.md](FIX_TASKS.md)). The PM reviewed your last work and found four issues that need correcting. Read the whole file first, then execute tasks in order.

Project context: single-cell 5G downlink simulator. DQN agent picks per-step UE allocation. Baselines: Random, RoundRobin, ProportionalFair. See [docs/summary.md](summary.md) for background.

Comments and identifiers stay terse, English. Default to no comments unless *why* is non-obvious.

---

## Task 1 — Fix the ActionAdapter properly (HIGH PRIORITY)

**Problem.** In the previous round you replaced `ActionAdapter.to_allocation` with the "simpler alternative" (all RBs → chosen UE). This is now structurally identical to ProportionalFair, and the comparison CSV at [results/baselines/comparison.csv](../results/baselines/comparison.csv) shows the DQN is **strictly worse than PF** in every scenario:

| Scenario | DQN Mbps | PF Mbps | DQN fairness | PF fairness |
|---|---|---|---|---|
| a | 60.4 | 70.5 | 0.10 | 0.10 |
| b | 75.7 | 114.8 | 0.067 | 0.067 |
| c | 60.7 | 162.2 | 0.05 | 0.05 |
| d | 34.4 | 81.1 | 0.05 | 0.05 |

The DQN's fairness is exactly `1/n_ue` — proof that it allocates every RB to one UE. This makes the project's headline RL claim hollow. Fix it.

**Required change** in [agents/train.py](../agents/train.py):

1. Replace `ActionAdapter` so that `action_dim` stays `n_ue` but allocation is per-RB and influenced by both the learned UE scores and channel quality.
2. Add a method to `DQNAgent` in [agents/dqn_agent.py](../agents/dqn_agent.py) called `q_values(state) -> np.ndarray` that returns the full Q-vector (size `n_ue`) instead of just `argmax`. Implementation: same as the inference path in `select_action`, but return `.cpu().numpy().flatten()` instead of the int.
3. New `ActionAdapter.to_allocation(snrs: np.ndarray, q_values: np.ndarray) -> np.ndarray` signature:
   - `snrs` is `state[:n_ue]` (SNR in dB per UE).
   - For each RB, compute a score per UE: `score[ue] = q_values[ue] + LAMBDA * snr_norm[ue]` where `snr_norm = snrs / 30.0` and `LAMBDA = 0.5`. Pick `argmax` per RB.
   - In this baseline form every RB ends up with the same UE, **but** add a tiny per-RB tiebreak noise from a fixed RNG (`np.random.default_rng(rb_idx)`) of magnitude `0.01` so that when two UEs are close in score, RBs split between them. This gives the DQN soft per-RB control.
   - Return shape `(n_rb,)` int64.
4. Update callers:
   - [agents/dqn_scheduler.py](../agents/dqn_scheduler.py) `allocate(state)`: call `q_values = self._agent.q_values(state)`, then `self._adapter.to_allocation(state[:self.n_ue], q_values)`.
   - In [agents/train.py](../agents/train.py) `AdaptedEnv.step`: change to take the full state and the agent. Cleanest: add the agent reference into `AdaptedEnv.__init__`, call `q_values = agent.q_values(state)` inside `step`, and pass both into `to_allocation`. The training loop already has both `state` and `agent` in scope — refactor `AdaptedEnv` so it stores `agent` and looks up `q_values` on each step. The DQN's `select_action(state)` is still used by the loop to choose the **stored action** (the int), and the env is given the **allocation** derived from `q_values(state)`.
   - Same pattern in [main.py](../main.py)'s `AdaptedEnv` (the inline class inside `run_training`).

5. Update [tests/test_adapter.py](../tests/test_adapter.py):
   - Remove the assertion `assert np.all(alloc0 == 0)`.
   - Add `test_action_adapter_per_rb`: build `q_values = np.zeros(10); q_values[3] = 1.0`, `snrs = np.full(10, 15.0)`, call `to_allocation(snrs, q_values)`, assert that **at least 90% of RBs go to UE 3** (the dominant pick) but **at least one RB goes to a different UE** (proves tiebreak noise gives soft splitting).
   - Add `test_action_adapter_snr_aware`: build `q_values = np.zeros(10)` (uniform), `snrs[0] = 25.0` (best), other UEs `snrs = 5.0`. Confirm UE 0 wins the majority of RBs.

**Verify.** `pytest tests/` passes 6 tests. The DQN's per-step fairness in the next baseline run will no longer equal exactly `1/n_ue`.

---

## Task 2 — Retrain scenario A and verify stability

**Problem.** Round 1 added Double DQN to [agents/dqn_agent.py:205-208](../agents/dqn_agent.py:205) but you never retrained to verify the loss-spike fix. The existing [results/logs/train_scenario_a.csv](../results/logs/train_scenario_a.csv) still shows max loss = 512,327.

**Required.** After Task 1 is complete (so the new adapter is in place):

1. Run `python main.py --scenario a --episodes 500`. This trains for 500 episodes only — keep it short.
2. The new log file will overwrite `results/logs/train_scenario_a.csv`. **Before training, copy the existing file to `results/logs/train_scenario_a_OLD.csv`** so the original record is preserved.
3. After training completes, read the new CSV and compute max loss. **Required: max loss < 100.**
4. If max loss >= 100, additionally apply: lower the learning rate from `5e-4` to `2.5e-4` in [configs/scenario_a.yaml](../configs/scenario_a.yaml) and **switch the loss function from `nn.MSELoss()` to `nn.SmoothL1Loss()`** (Huber loss) in [agents/dqn_agent.py:156](../agents/dqn_agent.py:156). Then retrain. Don't iterate more than twice.
5. Replace the new `model_final.pth` for scenario_a with this freshly trained one. Keep B/C/D models untouched (they will be retrained in Task 4).

**Verify.** Print max loss from the new `train_scenario_a.csv`. Required value: < 100.

---

## Task 3 — Revert unrequested GUI changes

**Problem.** Round 1 modified [gui/app.py](../gui/app.py) — button row layout split into two rows, speed scale rewritten with `_on_speed` callback, `_speed_disp` StringVar added. None of that was on the brief, and the brief said *"Do not refactor the GUI or add new features."*

**Required.** Revert only the GUI changes from your previous commit, leaving everything else intact.

```
git checkout HEAD -- gui/app.py
```

If `git checkout` would discard other intended changes (it shouldn't — `gui/app.py` is the only file affected), use `git diff HEAD -- gui/app.py` first to confirm scope. The original [gui/app.py](../gui/app.py) had a single `btn_f` frame holding all four buttons and a simpler `ttk.Scale` bound directly to `_speed_var`. Restore that.

**Verify.** `git diff HEAD -- gui/app.py` produces empty output.

---

## Task 4 — Retrain B/C/D under the new reward scale, OR add a load-time scale warning

**Problem.** In Round 1 you changed reward normalization in [env/network_env.py](../env/network_env.py) (`/100` removed, replaced with `THROUGHPUT_NORM_MBPS = 50.0` inside `_compute_reward`). But the existing checkpoints `results/models/scenario_{b,c,d}/model_final.pth` were trained on the **old** scale. Q-values inside them are off by ~50× from what the current training pipeline produces. Anyone resuming training from those checkpoints gets corrupted updates.

**Pick ONE option** based on time budget:

### Option A (preferred if cheap) — retrain B/C/D for 500 episodes each
For each scenario in `[b, c, d]`:
1. Copy old log to `train_scenario_{X}_OLD.csv` first.
2. Run `python main.py --scenario X --episodes 500`.
3. New `model_final.pth` overwrites the old one.
4. Confirm max loss < 100 (Task 2 fixes will be in place).

### Option B (fallback) — add explicit warning in `DQNAgent.load`
At the top of `DQNAgent.load` ([agents/dqn_agent.py:243](../agents/dqn_agent.py:243)), add:
```python
import warnings
warnings.warn(
    "Loaded checkpoint may have been trained on a different reward scale. "
    "If resuming training, verify env._compute_reward matches the training-time scale "
    "(current: throughput / THROUGHPUT_NORM_MBPS = 50). "
    "Pure inference (DQNScheduler) is unaffected.",
    RuntimeWarning,
)
```
Also add a one-line note to [results/models/TRAINING_NOTES.md](../results/models/TRAINING_NOTES.md) documenting which checkpoints predate the reward-scale change.

Choose Option A if all three retrains finish in under 10 minutes total on this machine; otherwise Option B. Document your choice in the final report.

---

## Task 5 — Re-run baseline comparison

After Tasks 1-4 are done:

1. Copy `results/baselines/comparison.csv` → `results/baselines/comparison_OLD.csv`.
2. Run `python scripts/run_baselines.py`.
3. Confirm new CSV exists with 16 rows.
4. **Sanity check the new DQN numbers:** DQN fairness should no longer equal exactly `1/n_ue` for all scenarios (that was the symptom of the broken adapter). At least one scenario must show fairness > `1/n_ue + 0.05`.

If the sanity check fails, the new ActionAdapter is still effectively single-UE — investigate whether `LAMBDA` and tiebreak noise are actually causing per-RB splits at runtime. Add a print statement temporarily to verify the allocation has multiple distinct values.

---

## Out of scope — do NOT do these

- Do not modify `gui/app.py` beyond the revert in Task 3.
- Do not introduce new dependencies.
- Do not commit, push, or open PRs.
- Do not retrain scenario A for more than 500 episodes (or B/C/D in Option A).
- Do not add new config files or scenarios.
- Do not write new docs except for the one-line note in Task 4 Option B.
- Do not "improve" the run_baselines.py script. Only re-run it.

---

## Final report

Print to stdout:
1. Which tasks completed (1-5 by ID), with a one-line note per task on what you did.
2. For Task 2: the actual max loss from the retrained scenario A log.
3. For Task 4: which option you chose and why.
4. For Task 5: the new DQN fairness values per scenario, and a one-line verdict on whether the adapter fix worked.
5. Output of `pytest tests/` (one line).
6. List of files modified.

Keep under 400 words.
