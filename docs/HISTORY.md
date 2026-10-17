# Project history

## v1.0: CSE 476 course submission (May 2026)

The project began as a three-person course project for *CSE 476 Mobile
Communication Networks*. It aimed to show that a Deep Q-Network can allocate
5G downlink resource blocks better than classical schedulers. The submitted
state is preserved under the git tag
[`v1.0-course-submission`](https://github.com/breksos/DRL-Based-5G-Simulation/tree/v1.0-course-submission).

The team's final report for that version reached an honest negative result:
**the DQN did not beat any baseline**. Its throughput and fairness were
indistinguishable from the random scheduler.

## The v2 review (October 2026)

Before the project was published as a portfolio piece, it was reviewed from
four angles: product scope, domain correctness, software architecture and QA.
The review found why the agent could not learn, plus several problems that
made the baselines and the reported numbers misleading.

### Why the v1 agent could not learn

| Problem | Evidence | v2 fix |
|---|---|---|
| **The stored action was not the executed action.** The ε-greedy action index went into the replay buffer, but the environment received an allocation sampled from `softmax(Q)`, which ignored that index. | Q-values differed across UEs by 0.3 on a mean of 54; the softmax entropy was 0.997 of its maximum (uniform random); corr(Q, SNR) = −0.02. | The agent builds each TTI one RB group at a time. Every decision it stores is the decision that was applied ([METHODOLOGY §4](METHODOLOGY.md#4-the-dqn-scheduler)). |
| **Q-values diverged** in three of the four scenarios. | Mean Q of 123–519, against a ceiling of 1/(1−γ) = 100 for rewards ≤ 1. | Rewards are increments of a bounded objective that telescope over a finite horizon (γ = 1). Huber loss and gradient clipping are kept. |
| **The problem had no sequential structure** (static channel, per-step reward), so γ = 0.99 only inflated Q. | | The sequential RBG formulation gives the episode real structure: each decision changes what the next one should be. |

### Why the v1 comparison was misleading

| Problem | v2 fix |
|---|---|
| Proportional fair updated the average only for the chosen UE, so **63–66 % of UEs got nothing for the whole episode**. Its "2× throughput" came from this bug. | Textbook PF: every UE's average is updated each TTI, with unscheduled UEs averaging in a zero. |
| SNR was clamped at 0 dB, making every UE beyond about 385 m (a quarter of them) look identical and overstating their rate by about 1.5×. | No clamp; the cell edge is at −4.3 dB. |
| UEs were dropped uniformly in *distance*, which over-represents UEs near the gNB. | Uniform over the cell *area*. |
| Two evaluation scripts gave conflicting numbers; the committed plots came from a stale run with the buggy PF. | One evaluation pipeline with common random numbers, 500 drops per scenario, and 95 % CIs. |
| Training and part of the evaluation were unseeded. | Seeded end to end; training determinism is tested. |
| Once optimised for real, the objective `α·T + β·J` turned out to **reward starving cell-edge UEs**. | A minimum-service penalty `λ·S` (with an ablation that keeps λ = 0). |

### Engineering

| v1 | v2 |
|---|---|
| 12 runnable entry points, 13 `sys.path` hacks; 3 scripts (5 commands) crashed | one `drl5g` CLI from an installable package (`pyproject.toml`, `src/` layout) |
| The env adapter was copy-pasted 5 times (3 copies broken) | one task class shared by training and inference |
| Defaults in 4 places; unknown config keys silently ignored | typed, frozen config dataclasses that reject unknown keys; each checkpoint embeds its config |
| Committed configs did not describe the committed models | a test asserts each model's embedded config equals the committed YAML |
| Test coverage 15.6 %, no CI | 96 % line coverage (GUI excluded), ruff, GitHub Actions on Python 3.10 and 3.12 |
| Mixed Turkish/English code, prints everywhere | English throughout, `logging` in library code |

The six overlapping documents of v1 (plan, summary, final report, integration
notes, user manual, project README) were replaced by the README,
[METHODOLOGY.md](METHODOLOGY.md) and [RESULTS.md](RESULTS.md). The originals
are still available under the v1 tag.

## What did not change

The research question, the four scenarios, the channel constants, the reward
weights α = β = 0.5, the baseline set (extended with Max C/I and a greedy
reference), the DQN family (Double DQN, Huber loss, target network, replay)
and the look of the live GUI.
