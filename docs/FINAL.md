# 5G DRL Resource Allocation — Final Report

---

## 1. What This Project Is

A simplified single-cell 5G downlink simulator where a Deep Q-Network (DQN) learns to
allocate per-step Radio Block (RB) assignments across multiple User Equipments (UEs). Three
classical schedulers — Random, Round Robin, and Proportional Fair — serve as baselines. All
policies are evaluated with a single combined-reward metric that weights total throughput
and Jain's fairness equally (α = β = 0.5). The codebase is a proof-of-concept: one gNodeB,
no inter-cell interference, path-loss-only channel model, and a discrete action space over
50–100 RBs.

---

## 2. Final State

**Code:** `env/`, `agents/`, `baselines/`, `evaluation/`, `gui/`, `scripts/`, `tests/`,
`configs/`, `utils/`. Four active scenarios (A/B/C/D). 11 tests passing.

**Trained models:** Per-scenario `model_final.pth` checkpoints under
`results/models/scenario_{a,b,c,d}/`, all produced under R3 specifications (2000 episodes,
lr = 2.5e-4, Double DQN + Huber loss, α = β = 0.5).

**Reward formulation:**
```
R = 0.5 · min(throughput_total / 50 Mbps, 1.0) + 0.5 · Jain_fairness
```
Clamped per step so rewards stay in [0, 1].

**Combined-R results (α = β = 0.5, mean over 5 seeds × 50 episodes per cell):**

| Scenario | Random R | RR R | PF R | DQN R | Best |
|---|---|---|---|---|---|
| A (10 UE / 50 RB)  | 0.5772 | 0.6074 | 0.6539 | 0.5764 | **PF** |
| B (15 UE / 75 RB)  | 0.7133 | 0.7497 | 0.6520 | 0.6645 | **RR** |
| C (20 UE / 100 RB) | 0.7720 | 0.8111 | 0.6486 | 0.6328 | **RR** |
| D (20 UE / 50 RB)  | 0.5299 | 0.5898 | 0.6334 | 0.4190 | **PF** |

**Verdict:** **The DQN does not beat any classical baseline on combined R in any of
the four scenarios.** The earlier "DQN beats PF in 3/4 scenarios" result was a
combined artifact of (i) a strawman PF implementation that gave all RBs to a single
argmax UE per step (PF fairness ≈ 0.10) and (ii) the throughput clamp at 50 Mbps,
which discards PF's actual throughput advantage. With per-RB PF and 5-seed
evaluation, PF wins A and D, RR wins B and C, and the DQN ranks last in A, C, D
and second-to-last in B. We report this as a negative result on the methods side
and discuss the architectural reasons in §5 and §6.

---

## 3. What Worked

### Round-Robin scheduler bug fix
**Problem:** The original RR counter was never advanced — every call assigned from UE 0.
**Fix:** Added `self._counter += 1` after each RB assignment.
**Impact:** RR fairness jumped from ~0.10 to 0.60–0.64, making it the correct fairness
upper-bound baseline. All subsequent DQN comparisons are now meaningful.

### Reward normalisation and clamping
**Problem:** Raw throughput in bps produced rewards in the tens of thousands, making
Q-value targets unstable and loss explode (max loss > 500,000 in early runs).
**Fix:** Normalise to Mbps and clamp: `norm_tp = min(tp_Mbps / 50, 1.0)`. Step reward now
lives in [0, 1].
**Impact:** Max training loss fell from 500,000+ to < 10. Training converged reliably.

### Double DQN + Huber loss
**Problem:** Standard DQN with MSE loss overestimated Q-values and diverged on scenarios B
and C.
**Fix:** Target computed via Double DQN rule (online network selects action, target network
evaluates). Huber loss replaces MSE.
**Impact:** Scenario B max loss: 325 → 4 (after also halving LR). Consistent convergence
across all four scenarios.

### Per-UE softmax-multinomial action adapter
**Problem:** Original adapter assigned all 50 RBs to a single argmax UE — no diversity.
**Fix:** For each RB, independently sample a UE from a softmax distribution over
`LAMBDA * SNR + (1 - LAMBDA) * Q_values`. Single seeded RNG for reproducibility.
**Impact:** Allocation now spreads across multiple UEs. DQN fairness rose from ~0.10 to
0.27–0.57 depending on scenario.

### Per-RB Proportional Fair (R7)
**Problem:** The previous PF was a strawman — it assigned *all* `n_rb` RBs to a single
argmax UE per step. Per-step Jain was therefore at most 1/n_ue (~0.05–0.10 across
scenarios), which made PF look like a low-fairness baseline that the DQN could beat
once the throughput term was clamped.
**Fix:** Standard per-RB PF — for each RB independently, pick `argmax(instant_rate / avg_rate)`,
then EMA-update only the chosen UE's avg_rate before allocating the next RB.
**Impact:** PF's mean per-step fairness rises from ~0.05–0.10 to ~0.27–0.31 across
scenarios. PF is now the combined-R winner in scenarios A and D and is competitive
in B and C. This invalidates the prior "DQN beats PF" claim — see §2 for the
corrected numbers and §4d for the strawman analysis.

### Test infrastructure
11 tests across `test_baselines.py` and `test_adapter.py` covering: RR rotation, PF lazy
init, per-RB PF distribution, per-RB PF avg-rate update locality, seeded randomness,
finite evaluation, reward clamping, multi-seed comparison-CSV schema, and softmax-adapter
distribution. These caught the RR bug, the PF init bias, the PF strawman, and adapter
edge cases during development.

### Reproducibility
`RandomScheduler`, `DQNScheduler`, and `ActionAdapter` all accept a `seed` argument.
`scripts/run_baselines.py` uses seed = 0 everywhere, making the 16-row `comparison.csv`
fully deterministic.

---

## 4. What We Tried That Didn't Work

### 4a. Tweaking LAMBDA + extending exploration (R4)

**Hypothesis:** Lowering the SNR weighting in the adapter (LAMBDA 0.5 → 0.1) would let the
DQN learn fairness-aware allocation rather than always favouring strong-channel UEs. Combined
with a longer epsilon-decay (50k → 200k steps) to allow more exploration, the agent might
discover more equitable policies.

**Result:** Marginal change. Combined-R shifts were ≤ 0.02 per scenario. DQN still failed
to beat Round Robin in any scenario after 2000 episodes. Training artefacts are logged in
`results/logs/train_scenario_*_LRhalf*.csv`.

**Why it failed:** The softmax-multinomial adapter makes each RB assignment an independent
draw. No matter how LAMBDA is tuned, the per-RB independence prevents the adapter from
reproducing Round Robin's deterministic uniform rotation. The bottleneck is architectural,
not a hyperparameter.

### 4b. Reward-weight shift to α = 0.7, β = 0.3 (R5)

**Hypothesis:** Throughput-favoring weights would let the DQN exploit its channel-awareness.
With α = 0.7, the agent should converge toward PF-like behaviour while still retaining some
fairness signal.

**Result:** Net regression vs R3. Proportional Fair dominated wherever its throughput hit
the clamp ceiling (all four scenarios at high UE counts), because the 0.7 weight amplifies
PF's throughput advantage while its near-zero fairness is discounted. DQN lost to PF in 2/4
scenarios (previously 1/4 at α = β = 0.5). No scenario improved relative to R3.

**Why it failed:** The throughput term clamps at `min(tp/50, 1.0)`. Any scheduler that
pushes total throughput above 50 Mbps gets the same throughput contribution regardless of
α. Raising α does not reward throughput gains beyond the ceiling — it only penalises
fairness harder. PF consistently exceeds 50 Mbps in scenarios B, C, D, making α = 0.7 net
harmful.

### 4c. Domain-randomized warm-start training (R6)

**Hypothesis:** Training one model on randomized cell radii (Uniform[400, 800] m) and
per-episode mobility (Bernoulli(0.5)), warm-started from the R3 scenario-A checkpoint with
ε = 0.3, would produce a policy that transfers to unseen environments. A held-out scenario E
(750 m cell, mobility ON) was constructed as the transfer test.

**Setup:** 2000 warm-start episodes, lr = 2.5e-4, all other hyperparameters matching
scenario A. Scenario E evaluated on 50 episodes for all four schedulers plus the general
model.

**Result:** DQN(general) combined R on scenario E = 0.4661. DQN(scenario_a R3) on scenario
E = 0.4671. Transfer delta = −0.001. The threshold for a meaningful result was +0.02.
Training log: `results/logs/train_scenario_general.csv` (retained as audit).

**Why it failed:** The per-RB softmax adapter uses SNR values directly as part of the
logits. The learned Q-values therefore encode SNR magnitudes typical of the training
distribution (500 m cell, mostly 5–20 dB). A 750 m cell shifts the SNR distribution
downward by ~4 dB. The adapter's combined logit `LAMBDA * SNR + (1-LAMBDA) * Q` degrades
gracefully in throughput but the fairness pattern is unchanged — the policy has not learnt
any cell-radius-invariant allocation principle. 2000 warm-start episodes with ε = 0.3 is
not enough to override the R3 prior.

### 4d. Strawman PF (discovered post-hoc)

**The bug.** The original `ProportionalFairScheduler.allocate` computed
`argmax(instant_rate / avg_rate)` once per step and assigned *all* `n_rb` RBs to that
single UE: `return np.full(n_rb, best, dtype=int)`. This is not Proportional Fair — it
is "winner-takes-all PF." Per-step Jain's index for any 4-/10-/15-/20-UE cell could
not exceed 1/n_ue (≈ 0.05–0.10).

**Why it inflated DQN's apparent performance.** Combined R = 0.5 × min(tp/50, 1) + 0.5 × J.
With strawman PF: PF's tp far exceeds 50 Mbps so its throughput term saturates at 1.0
*and* its fairness contribution is ≈ 0.10 → combined R ≈ 0.55. The DQN's softmax-multinomial
adapter naturally spreads RBs and reaches J ≈ 0.27–0.57, which on the same clamped throughput
beats 0.55 in scenarios A/B/C. So "DQN beats PF" was really "any scheduler with non-degenerate
fairness beats this PF," not a property of the learned policy.

**The fix and its effect.** The correct per-RB PF (each RB independently picks
`argmax(instant_rate / avg_rate)` and EMA-updates the chosen UE's avg_rate before the
next RB) raises PF's mean per-step fairness to ~0.27–0.31. PF then wins scenarios A
and D outright on combined R; the DQN loses to all three classical baselines in three
of four scenarios. The bug existed before R3 and was carried through every prior round.

---

## 5. Honest Limitations

**The DQN does not beat any classical baseline on combined R.** Previous rounds
reported "DQN beats PF in 3/4 scenarios." That result was a combined artifact of
the reward clamp at 50 Mbps and the strawman PF described in §4d. With per-RB PF
and 5-seed evaluation, the DQN is last in scenarios A, C, D and second-to-last in B.
There is no scenario where the DQN improves over both PF and RR.

**DQN never beats Round Robin.** RR's deterministic uniform rotation achieves fairness
0.60–0.64. The softmax-multinomial adapter is fundamentally stochastic per RB — it cannot
deterministically rotate. The maximum fairness the adapter achieves is ~0.57 (scenario A),
and it averages 0.25–0.57 across scenarios. Beating RR would require a policy architecture
that can output a deterministic rotation or an action space that assigns UE slices rather
than per-RB assignments.

**The Bellman target regresses onto the wrong action.** The DQN stores a single integer
in the replay buffer per step (the argmax UE), but the reward is produced by the
per-RB softmax-multinomial allocation. Q-learning therefore regresses Q(s, ue*) toward
a return generated by an allocation that was not "pick UE ue* once" — it was a
50-RB stochastic spread. The principled fix is a per-(UE, RB) Q-network with a
deterministic argmax-per-RB head (see §6, Direction 1).

**Per-scenario models only.** Scenarios B, C, D have different state dimensions (75, 100,
or 70 features) from scenario A (70 features). No model transfers across scenarios with
different UE or RB counts.

**Compute budget.** All training ran on CPU at 2000 episodes per scenario. Scenario A was
run at 3676 episodes in earlier rounds. Longer training (10k+ episodes) or GPU parallelism
might narrow the gap to RR, but there is no theoretical guarantee given the adapter's
structural limitations.

**Simulation fidelity.** Single cell, one gNodeB, path-loss-only channel model, no
frequency-selective fading, no inter-cell interference, no traffic queues, no modulation
and coding scheme tables. The results are proof-of-concept demonstrations of the MDP
formulation and DQN architecture, not production-grade evaluations.

---

## 6. What Future Work Would Look Like

**Direction 1 — Per-(UE, RB) Q-network (medium effort)**
Replace the per-UE softmax adapter with a Q-network that outputs a score for each
(UE, RB) pair, then greedily assigns each RB to its top-scoring UE. This preserves the DQN
framework but makes the allocation deterministic conditioned on Q-values. In principle, the
network could learn to reproduce RR's uniform rotation by assigning equal scores to all
(UE, RB) pairs. The state would remain the same 70-feature vector; the action head would
expand from `n_ue` outputs to `n_rb × n_ue` outputs. Expected effort: 1–2 weeks of
architecture changes and retraining.

**Direction 2 — Variable-size input with masked attention (high effort)**
To get a single model that transfers across scenarios with different UE and RB counts,
replace the fixed-size MLP with a transformer or attention mechanism that treats each UE as
a token and each RB as a query. Masking handles variable counts. Training on randomized
(n_ue, n_rb, cell_radius) tuples would produce a genuinely generalisable scheduler. This is
the only clean path to the "one model, all scenarios" goal. Expected effort: 1–2 months
including simulation harness changes.

**Direction 3 — Richer simulator (high effort)**
Move to ns-3 with the 5G-LENA module or simuLTE to introduce frequency-selective fading,
inter-cell interference, real modulation-and-coding-scheme (MCS) tables, and traffic
buffers. In these environments, PF's heuristic starts to break down under bursty traffic
and non-stationary channels, giving DRL a genuine advantage. This is the environment where
the approach's value becomes empirically defensible. Expected effort: several months of
simulator integration.

---

## 7. Repository Map

```
5G Simulation/
├── agents/           DQN agent, training loop, action adapter, DQN scheduler
│     ├── dqn_agent.py       Double DQN, Huber loss, experience replay
│     ├── train.py           Training loop + ActionAdapter (softmax-multinomial)
│     └── dqn_scheduler.py   Wraps DQNAgent for baseline-compatible interface
├── baselines/        Classical schedulers (Random, RoundRobin, ProportionalFair)
├── configs/          Per-scenario YAML files (scenario_a through scenario_d, default)
├── env/              Gymnasium environment (network_env.py) + channel model
├── evaluation/       evaluate_scheduler() metrics helper
├── gui/              Real-time Tkinter dashboard  →  python main.py --gui
├── scripts/
│     ├── run_baselines.py   Evaluate all schedulers on all scenarios
│     │                      Output: results/baselines/comparison.csv (16 rows)
│     └── train_*.py         Per-scenario training scripts
├── tests/            11 pytest tests (test_baselines.py, test_adapter.py)
├── utils/            Config loader
├── results/
│     ├── models/     Per-scenario model_final.pth checkpoints + audit copies
│     ├── logs/       Training CSV logs (episode reward, loss, epsilon)
│     └── baselines/  comparison.csv + historical comparison_*OLD.csv audit files
├── docs/
│     ├── FINAL.md          This file — post-mortem and lessons learned
│     ├── summary.md        Presentation-ready summary with results table
│     ├── BERK_INTEGRATION.md   Turkish status snapshot for the integration reviewer
│     ├── plan.md           Original project plan
│     ├── USER_MANUAL.md    End-user GUI and script guide
│     └── source/           Project proposal PDFs (untouched)
└── main.py           Entry point:  --demo (10-step run)  |  --gui (dashboard)
```

**Quick start:**
```bash
python main.py --demo                  # sanity check, no deps beyond numpy/gymnasium
python scripts/run_baselines.py        # reproduce the 16-row comparison table
python main.py --gui                   # live dashboard
python -m pytest tests/                # 11 tests
```
