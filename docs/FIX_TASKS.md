# Developer Agent Task List — 5G DRL Project Fixes

You are a developer agent (Sonnet 4.6, medium effort) working on the **5G DRL Resource Allocation** project at the repo root `C:\Users\ASUS\Desktop\Projeler\5G Simulation`. A PM review found the issues below. Read the whole file first, then fix items in the order listed. Each task lists the **file/line**, the **problem**, and the **required fix**. Do not invent extra refactors. After each task, verify behaviour with the listed check.

Project context: a single-cell 5G downlink simulator (`env/`), a DQN agent (`agents/`), three baselines (`baselines/`), evaluation utilities (`evaluation/`), a Tkinter/matplotlib GUI (`gui/`), and YAML scenario configs (`configs/`). Reward = `α·throughput_Mbps + β·Jain_fairness`. See [docs/summary.md](summary.md) and [docs/plan.md](plan.md) for background.

Comments and identifiers may be in Turkish — keep new comments terse and in **English**. Default to no comments unless the *why* is non-obvious.

---

## Priority 1 — Correctness bugs

### 1.1 Round-Robin scheduler never rotates between steps
**File:** [baselines/schedulers.py:38-44](../baselines/schedulers.py)
**Problem:** `self._counter = (self._counter + self.n_rb) % self.n_ue`. With defaults `n_rb=50, n_ue=10`, this is `+50 % 10 = 0`, so the starting offset never advances. Every step produces identical allocation `[0,1,…,9,0,1,…]` — there is no temporal rotation across steps, defeating the purpose of RR.
**Fix:** Increment by 1 each step: `self._counter = (self._counter + 1) % self.n_ue`.
**Check:** Call `allocate(state)` 3 times in a row with `n_ue=10, n_rb=50` and confirm the first element is 0, 1, 2 on successive calls.

### 1.2 ActionAdapter cripples the DQN's action space
**File:** [agents/train.py:20-41](../agents/train.py)
**Problem:** Current adapter forces the DQN-chosen UE to take the first half of all RBs, with the remaining half hard-coded round-robin. The DQN effectively has only `n_ue` distinct policies and 25 of 50 RBs are never under its control. This contradicts [docs/summary.md:82-86](summary.md) which advertises per-RB allocation.
**Fix:** Replace `ActionAdapter` with a **per-UE scoring** approach:
- `action_dim` stays `n_ue` (the DQN outputs one Q-value per UE — this is the UE's "priority score" for the current state).
- Replace `to_allocation(dqn_action: int)` with `to_allocation(state, q_values: np.ndarray) -> np.ndarray` that assigns each RB to the UE with the highest score, *adjusted by SNR*: for each RB pick `argmax(q_values * snr_capacity_per_ue)`. This ensures every RB is influenced by the learned policy AND by channel quality.
- Simpler alternative if the above is too invasive: keep the int-action interface but allocate **all** RBs to the chosen UE (pure greedy). Document whichever choice is taken.
- Update `select_action` callers in [agents/train.py](../agents/train.py), [agents/dqn_scheduler.py](../agents/dqn_scheduler.py), and [main.py](../main.py) accordingly. If you adopt the q-values version, expose a method on `DQNAgent` that returns the full Q-vector (not just argmax) for the adapter to consume.

**Important:** if you change the adapter signature, also update `DQNScheduler.allocate` ([agents/dqn_scheduler.py:60-68](../agents/dqn_scheduler.py)) and the `AdaptedEnv` wrappers in [agents/train.py:201-211](../agents/train.py) and [main.py:79-81](../main.py).

**Check:** With a fresh untrained agent and `n_rb=50`, two consecutive `allocate(state)` calls on **different** states should produce **different** allocations (not just the first half flipping). At least 80% of RBs should differ when SNR vector is permuted.

### 1.3 Reward normalization is asymmetric (training vs reporting)
**File:** [agents/train.py:114](../agents/train.py)
**Problem:** Buffer stores `reward / 100.0` but logs and downstream comparisons use raw reward. Q-values learned are on the /100 scale, but [docs/summary.md:196-204](summary.md) compares raw rewards.
**Fix:** Pick **one** scale and stick to it. Recommended: normalize once inside the env (`_compute_reward` in [env/network_env.py:117-120](../env/network_env.py)) by dividing total throughput by a constant `THROUGHPUT_NORM_MBPS = 50.0` so that reward stays roughly in `[0, 2]`. Then remove the `/100.0` in `train.py`. Update the `info["throughput_total"]` field to remain in **bps** (unchanged) so evaluation metrics in [evaluation/metrics.py](../evaluation/metrics.py) keep working.
**Check:** Episode reward over 200 steps should be in roughly `[100, 400]` range, not thousands. CSV log should match.

### 1.4 PF scheduler initialisation is degenerate
**File:** [baselines/schedulers.py:54-70](../baselines/schedulers.py)
**Problem:** `_avg_rates` initialised to `1.0` (dimensionless) while `instant` capacity is in **bps (~10⁶)**. First call: priority is ~10⁶ for every UE → `argmax` deterministically returns UE 0 until the EMA warms up. PF results are biased toward UE 0 for many initial steps.
**Fix:** Lazy-init `_avg_rates` on the first `allocate` call: set it to `instant.copy()` (or `instant.mean() * np.ones(n_ue)`) so priorities start at ~1.0 across the board. Reset to "uninitialised" sentinel (e.g. `None`) inside `reset()`.
**Check:** With distances `[100, 100, 100, …]` (uniform SNR) the first 10 calls should not all return the same UE.

### 1.5 Mobility model contradicts documentation
**Files:** [env/network_env.py:58-64](../env/network_env.py) and [docs/summary.md:94-95](summary.md)
**Problem:** Docs say *"After 200 steps, UE positions reset randomly."* But when `mobility=True`, only the first reset is random; every subsequent reset does only ±10 m walk — never a full re-randomisation.
**Fix:** Decide which behaviour is intended and reconcile.
- If mobility means *continuous drift across episodes*: update [docs/summary.md:94-95](summary.md) to say "with mobility=True, UE positions drift ±10 m per episode reset; with mobility=False they re-randomise each episode."
- If mobility means *random walk within an episode*: move the walk into `step()`, not `reset()`.
The simpler choice is the first (doc-only fix).
**Check:** Run 5 resets with mobility on and confirm `env.distances` evolves smoothly; with mobility off confirm it changes drastically each reset.

---

## Priority 2 — Training & evaluation integrity

### 2.1 Q-loss explodes during training
**File:** [results/logs/train_scenario_a.csv](../results/logs/train_scenario_a.csv) shows loss=107477 at episode 3673 with epsilon already at the floor.
**Fix:** Apply **Double DQN** target computation in [agents/dqn_agent.py:205-207](../agents/dqn_agent.py):
```python
with torch.no_grad():
    next_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
    max_next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
    targets = rewards_t + self.gamma * max_next_q * (1.0 - dones_t)
```
Also reduce `target_update_freq` from 1000 → 500 in [configs/default.yaml](../configs/default.yaml) and the four scenario configs.
**Check:** Retrain scenario A for 500 episodes. Max loss in CSV must stay below 100. (You don't need to retrain the full 10k — just demonstrate stability on a short run.)

### 2.2 No artefact backs the report's headline numbers
**Problem:** [results/baselines/](../results/baselines/) is empty. The comparison table in [docs/summary.md:196-204](summary.md) has no source CSV/JSON.
**Fix:** Create `scripts/run_baselines.py` that:
- Loads each scenario config (a, b, c, d).
- Runs Random, RoundRobin, ProportionalFair, and DQN (loaded from `results/models/scenario_X/model_final.pth`) for 50 episodes each via `evaluation.metrics.evaluate_scheduler`.
- Writes `results/baselines/comparison.csv` with columns `scenario, scheduler, avg_throughput_mbps, std_throughput_mbps, avg_fairness, std_fairness`.
- Uses `np.random.seed(0)` and an env seed for reproducibility.

**Check:** `python scripts/run_baselines.py` runs to completion and produces a non-empty CSV with 16 rows (4 scenarios × 4 schedulers).

### 2.3 Inconsistent training budgets across scenarios
**File:** training logs show A=3676 ep, B/C/D=2000 ep, configs say 10000.
**Fix:** This is documentation-only — do **not** retrain everything. Add a `results/models/TRAINING_NOTES.md` documenting actual episodes per scenario and noting the inconsistency for the report. Update each scenario's `n_episodes` in `configs/scenario_*.yaml` to the value actually trained, so future re-runs match.
**Check:** File exists with one row per scenario.

### 2.4 RandomScheduler is non-reproducible
**File:** [baselines/schedulers.py:24-28](../baselines/schedulers.py)
**Fix:** Accept an optional `seed` in `__init__` and use `self._rng = np.random.default_rng(seed)` instead of global `np.random`. Update `evaluate_scheduler` callers if needed.
**Check:** Two `RandomScheduler(n_ue, n_rb, seed=0)` instances produce identical allocations.

---

## Priority 3 — Code quality & cleanup

### 3.1 Remove MockEnv from production training module
**File:** [agents/train.py:48-66](../agents/train.py)
**Problem:** Phase-1 debug scaffolding lives in the same file the GUI imports. The `--mode mock` branch in the `__main__` block is no longer needed.
**Fix:** Delete `MockEnv` class and the `mock` branch in the argparse `__main__`. If you want to keep it, move it to `agents/_mock_env.py` (underscore prefix to mark internal) and don't import it by default.
**Check:** `python -m agents.train --mode 5g` still works; `--mode mock` is gone.

### 3.2 Defensive modulo masks bugs
**File:** [evaluation/metrics.py:28-29](../evaluation/metrics.py) and [env/network_env.py:81-85](../env/network_env.py)
**Problem:** `int(ue) % n_ue` silently corrects malformed actions instead of failing loudly. Combined with action_space=`MultiDiscrete([n_ue]*n_rb)`, this should never trigger — if it does, that's a bug worth surfacing.
**Fix:** Replace `int(ue) % n_ue` with `int(ue)` and add `assert 0 <= int(ue) < n_ue, f"action out of range: {ue}"` once at top of the function (validate the whole action array, not per-element in the loop).
**Check:** Existing tests still pass; passing an out-of-range action raises AssertionError.

### 3.3 Add minimal tests for the RL pipeline
**Problem:** Only the channel model has tests. The adapter, baselines, and evaluation are untested.
**Fix:** Create `tests/test_baselines.py` and `tests/test_adapter.py` with these cases:
- `test_round_robin_rotates`: confirm fix 1.1.
- `test_pf_no_initial_bias`: confirm fix 1.4.
- `test_random_seeded`: confirm fix 2.4.
- `test_action_adapter_uses_state`: confirm fix 1.2 (different states → different allocations).
- `test_evaluate_scheduler_finite`: end-to-end smoke that `evaluate_scheduler(env, RandomScheduler, n_episodes=2)` returns finite floats.

Use `pytest`. Keep each test under 15 lines. No fixtures unless needed.
**Check:** `pytest tests/` is green.

### 3.4 Stale top-level files referenced by git
**File:** repo root
**Problem:** `git status` shows `BERK_INTEGRATION.md` and `README_Project.md` deleted but not committed. `docs/` is untracked.
**Fix:** Do **not** commit anything yourself. Just verify those files are truly redundant (their content lives in `docs/` now) and add a note to your final report telling the user "ready to commit: 2 deletions + new docs/ folder". Let the user run the commit.

### 3.5 Mixed Turkish/English comments
**Files:** [agents/dqn_agent.py](../agents/dqn_agent.py), [agents/train.py](../agents/train.py), [agents/dqn_scheduler.py](../agents/dqn_scheduler.py), [main.py](../main.py)
**Fix:** Do **not** mass-translate. Only when you touch a function for another fix in this list, translate any Turkish comments inside that function to short English. Don't add comments that didn't exist.

---

## Out of scope — do NOT do these

- Do not retrain the full 10k-episode budget for any scenario (too expensive). Only do short retrains for verification (≤500 episodes).
- Do not refactor the GUI or add new features.
- Do not introduce new dependencies beyond what's in `requirements.txt`.
- Do not commit, push, or open PRs. Leave changes uncommitted for the user to review.
- Do not write README files, design docs, or changelogs unless the task list explicitly says so.

---

## Final report

After completing all tasks, write a brief summary to stdout containing:
1. Which tasks you completed (by ID).
2. Any tasks you skipped and why.
3. The output of `pytest tests/` (one line).
4. The output of `python scripts/run_baselines.py` (just confirm it produced the CSV — no need to paste contents).
5. A list of files modified.

Keep the summary under 300 words.
