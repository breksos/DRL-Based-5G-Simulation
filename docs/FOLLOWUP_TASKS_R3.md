# Developer Agent Task List — Round 3

You are a developer agent (Sonnet 4.6, medium effort) at the repo root `C:\Users\ASUS\Desktop\Projeler\5G Simulation`. This is round 3 of fixes after a QA tester audit. Read the whole file first, then execute tasks in order.

**Background.** The previous round successfully made the action adapter do per-RB allocation (softmax + multinomial). However a tester found the DQN still loses to PF and RR on the project's own combined-reward metric in every scenario. Root cause: the reward function in `env/network_env.py` divides throughput by 50 with no upper clamp, so PF's high-throughput / low-fairness allocations get unboundedly large rewards that drown out the fairness term. The reward function is structurally unbalanced; the DQN was never trying to learn the right thing.

Current numbers from [results/baselines/comparison.csv](../results/baselines/comparison.csv), with combined R = 0.5·(tp/50) + 0.5·fairness:

| Scenario | Random | RR | PF | DQN |
|---|---|---|---|---|
| a | 0.562 | 0.591 | 0.755 | 0.573 |
| b | 0.696 | 0.731 | 1.181 | 0.692 |
| c | 0.828 | 0.866 | 1.647 | 0.757 |
| d | 0.519 | 0.577 | 0.836 | 0.452 |

PF wins all four; DQN is last in three.

Comments stay terse, English. No new comments unless *why* is non-obvious. Do not commit. No new dependencies.

---

## Task 1 — Clamp the reward (BLOCKER fix)

**File:** [env/network_env.py:120-125](../env/network_env.py)

**Problem.** `_compute_reward` returns `α·(tp/50) + β·fairness` where the `tp/50` term is unbounded (PF reaches ~3.2 in scenario C). Reward is supposed to be roughly in `[0, 1]` per step, with throughput and fairness equally weighted. Right now the throughput term swamps fairness, making the agent's goal "maximize throughput at any cost" — same as PF's goal — but PF will always win that race because it's a hand-coded throughput-maximizer.

**Fix.** Clamp the normalised throughput to `[0, 1]` before applying the weight:
```python
def _compute_reward(self, throughputs: np.ndarray) -> float:
    norm_tp = min(throughputs.sum() / 1e6 / THROUGHPUT_NORM_MBPS, 1.0)
    fairness = self._jains_index(throughputs)
    return self.alpha * norm_tp + self.beta * fairness
```

After clamp, max reward per step = `0.5·1.0 + 0.5·1.0 = 1.0`, max episode reward = `200`. Both halves are now in `[0, 1]` and equally weighted as the spec promised.

**Verify.** Run `python main.py --demo` and confirm step rewards are now in `[0, 1]` (currently the demo shows raw rewards — fine, they'll just be smaller). Add a quick assert in [tests/test_baselines.py](../tests/test_baselines.py): `test_reward_is_clamped` — step the env once with an action that gives one UE all 50 RBs (action = `np.zeros(50, dtype=int)`), assert returned reward ≤ 1.0.

---

## Task 2 — Fix per-RB RNG bug

**File:** [agents/train.py:44-45](../agents/train.py)

**Problem.** The adapter loop creates `np.random.default_rng(rb_idx)` for each RB. This means RB index 0 always uses seed 0, RB index 1 always uses seed 1, etc. across every state and every episode. The samples are not actually independent — they are deterministic per (rb_idx, probs).

**Fix.** Construct one RNG per `to_allocation` call (not per RB). Drop the seed for runtime stochasticity, OR keep determinism by making `ActionAdapter.__init__` accept an optional `seed` and create `self._rng = np.random.default_rng(seed)` once. Replace the loop body with a single vectorised call:
```python
allocation = self._rng.choice(self.n_ue, size=self.n_rb, p=probs).astype(np.int64)
```

Default `seed=None` (true randomness). If callers want reproducibility they can pass `seed=0`. Update `scripts/run_baselines.py` to pass `seed=0` when constructing the `DQNScheduler` (and update `DQNScheduler` to forward the seed to the `ActionAdapter`).

**Verify.** Add `test_action_adapter_distribution` in [tests/test_adapter.py](../tests/test_adapter.py): with `q_values = np.array([1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0])` and uniform SNRs, assert that **at least 35** of 50 RBs go to UEs 0 or 1 (the high-priority pair). Use a fresh adapter with `seed=0`.

---

## Task 3 — Retrain all 4 scenarios

After Tasks 1 and 2, the reward landscape and adapter behaviour have both changed. Existing checkpoints are stale.

For each scenario in `[a, b, c, d]`:
1. Move existing log to `results/logs/train_scenario_{X}_R3OLD.csv` (don't overwrite previous OLD/OLD2 audit files).
2. Move existing `results/models/scenario_{X}/model_final.pth` to `results/models/scenario_{X}/model_final_R3OLD.pth`.
3. Run `python main.py --scenario X --episodes 2000`.
4. Confirm max loss in new log < 100. If max loss ≥ 100, retry with `lr` halved in the corresponding `configs/scenario_{X}.yaml` (don't iterate more than twice per scenario).

**Verify.** Print one line per scenario: `scenario X: max_loss = Y, last100_avg_reward = Z`.

---

## Task 4 — Re-run baselines and verify DQN wins on combined reward

1. Move `results/baselines/comparison.csv` → `results/baselines/comparison_R2OLD.csv`.
2. Run `python scripts/run_baselines.py`.
3. Read the new CSV and compute combined reward `R = 0.5·min(tp/50, 1.0) + 0.5·fairness` for each row.
4. **Required outcome:** DQN must beat **all three baselines** on combined R **in at least 3 of 4 scenarios**. If it fails in more than one scenario, raise this in the final report — do not iterate further (out of budget).

**Verify.** Print the combined-R table per scenario in the final report.

---

## Task 5 — Update the stale docs

### 5a. [docs/BERK_INTEGRATION.md](BERK_INTEGRATION.md)
Currently lists the adapter as "still single-UE" — that was true in round 2, false now. Update sections:
- "Durum Özeti" table: action adapter row is now ✅.
- "Açık sorunlar": remove the adapter issue (#1) entirely, and remove the per-RB RNG issue (#6) — both fixed in this round. Renumber remaining open items.
- Add a new dated entry under "Yapılanlar" for Round 3 work.
- "Audit trail" section: add R3OLD entries.

### 5b. [docs/summary.md](summary.md)
The "Expected Results" table at line ~196 is no longer aspirational — fill in the **actual** numbers from the new `comparison.csv` for both throughput, fairness, and combined R. Don't fabricate. If DQN still loses to PF on raw throughput, that's fine — the table now also has fairness and combined columns where DQN should win.

Specifically replace the existing table with one that has columns: `Scheduler | Throughput Mbps | Fairness | Combined R`. One row per scheduler, averaged across all 4 scenarios (or one table per scenario — your choice, whichever fits in 20 lines).

### 5c. Confirm Turkish/English consistency
Don't translate aggressively. Only normalize where it's confusing — e.g., a Turkish docstring on a function whose body now has English comments.

---

## Out of scope — do NOT do these

- Do not commit, push, or open PRs.
- Do not modify `gui/app.py` (the user has uncommitted changes there they're managing manually).
- Do not retrain for more than 2000 episodes per scenario.
- Do not introduce new dependencies.
- Do not refactor scheduler interfaces, env API, or GUI.
- Do not tune additional hyperparameters beyond LR halving fallback in Task 3.

---

## Final report

Print to stdout, under 500 words:

1. **Tasks completed** by ID, with one-line note each.
2. **Task 3 stability table:** scenario, episode count, max loss, last-100 reward.
3. **Task 4 result table:** scenario × scheduler grid with combined R values. Bold the winner per scenario (or just star it with `*`).
4. **Verdict:** does DQN now beat all three baselines on combined R in ≥ 3/4 scenarios? Yes / No.
5. `pytest tests/` output (one line).
6. List of files modified.
7. **Anything you noticed that wasn't on the task list** but a PM should care about (e.g., a baseline scheduler producing surprising numbers, a previously-hidden bug in env code, etc.). Keep this brief — flag, don't fix.
