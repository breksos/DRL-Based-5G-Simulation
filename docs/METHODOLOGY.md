# Methodology

This document describes the system model, the scheduling objective, the
schedulers and how they are evaluated. Every number here is taken from
[`configs/`](../configs) or [`src/drl5g`](../src/drl5g).

## 1. System model

One gNB serves `N` user equipments (UEs) over `B` resource blocks (RBs) of
180 kHz each. Time is divided into TTIs. In every TTI the scheduler assigns
each RB to exactly one UE.

### Channel ([`channel.py`](../src/drl5g/channel.py))

| Quantity | Model |
|---|---|
| Path loss | `PL(d) = 70.25 + 37.6 · log10(d)` dB, `d` in metres |
| Per-RB SNR | `SNR = 46 dBm − PL(d) − N0`, with `N0 = −174 dBm/Hz + 10·log10(180 kHz)` |
| Per-RB rate | Shannon: `C = 180 kHz · log2(1 + SNR)` |
| UE placement | uniform over the cell area, 50 m ≤ d ≤ 500 m, re-drawn every episode |

The path-loss intercept and transmit power are calibrated together to give a
useful spread of channel qualities: about 33 dB at 50 m, 22 dB at 100 m and
−4 dB at the 500 m cell edge, so the best UE gets roughly 25× the per-RB rate of
the worst. There is no shadowing or fast fading. A UE's channel is therefore
constant for a whole episode, and the problem is an allocation problem rather
than a channel-tracking one (see [Limitations](#6-limitations)).

### Scenarios ([`configs/`](../configs))

| ID | UEs | RBs | RBs per UE | RB group `g` | Character |
|---|---|---|---|---|---|
| A | 10 | 50 | 5 | 2 | baseline |
| B | 15 | 75 | 5 | 3 | medium load |
| C | 20 | 100 | 5 | 4 | dense, but enough spectrum to reach the throughput target easily |
| D | 20 | 50 | 2.5 | 2 | resource-scarce: the target is hard to reach while serving everyone |

The RB group size `g` only matters for the DQN and the greedy reference (§4).
It is chosen so that every TTI takes 25 decisions. Like NR's resource-block
groups, which are 2–16 RBs depending on the bandwidth part, it trades
scheduling granularity for decision cost.

## 2. Scheduling objective

Every scheduler is judged on the same per-TTI objective, which is also the
environment's reward ([`metrics.py`](../src/drl5g/metrics.py)):

```
R = α · min(T / T₀, 1)  +  β · J  −  λ · S
```

| Term | Meaning | Value |
|---|---|---|
| `T` | total cell throughput in the TTI | |
| `T₀` | cell service target; the throughput term saturates here | 50 Mbit/s |
| `J` | Jain's fairness index of the per-UE throughputs, `(Σx)² / (N·Σx²)` | |
| `S` | share of UEs that received no RB in the TTI | |
| `α, β` | weights | 0.5, 0.5 |
| `λ` | starvation (minimum-service QoS) penalty | 1.0 |

The throughput term saturates. Once the cell carries `T₀`, extra throughput
is worth nothing and only fairness can still raise `R`. That saturation is what
makes this a genuine trade-off: serving only the best UE (Max C/I) hits the cap
but has `J = 1/N`, while equal throughput for everyone needs most RBs to go to
cell-edge UEs and collapses `T`.

**Why the starvation term exists.** The original course objective had only
the first two terms (`λ = 0`). Once the agent could actually optimise it, the
optimum turned out to drop cell-edge UEs altogether. A UE given nothing barely
lowers Jain's index, while the RBs it would need raise throughput a lot
elsewhere. The [λ = 0 ablation](RESULTS.md#ablation-the-original-objective-rewards-starvation)
shows both the DQN and the objective-aware greedy heuristic doing this. The
penalty is a minimum-service requirement: every UE should get at least one RB
group in every TTI.

## 3. Schedulers

| Name | Rule | Role |
|---|---|---|
| **Random** | each RB to a uniformly random UE | floor |
| **RoundRobin** | deal RBs in turn, continuing where the last TTI stopped | equal resource share |
| **PropFair** | per RB, `argmax r_k / T_k`; `T_k` is an EMA (window 100 TTIs) of served throughput, updated for every UE (unscheduled UEs average in a zero) | textbook proportional fair |
| **MaxRate** | every RB to the best channel (Max C/I) | throughput ceiling |
| **Greedy** | per RB group (same `g` as the DQN), the UE that raises `R` the most right now | model-based reference: it knows `R` in closed form |
| **DQN** | learned, see below | the method under study |

With a static channel, proportional fair converges to equal RB shares, which is
exactly round robin. There is no fading for it to exploit, and the identical
results for the two in the evaluation are expected.

## 4. The DQN scheduler

### 4.1 Decision process ([`rl/allocation.py`](../src/drl5g/rl/allocation.py))

Choosing all `B` assignments at once means an action space of `N^B` (10⁵⁰ in
scenario A). Instead the agent **fills one TTI decision by decision**. Each
step gives the next *resource block group* (RBG) of `g` RBs to one UE, so a TTI
takes `B / g = 25` decisions and the action space is just `N`.

In a pilot on scenario A, per-RB decisions (`g = 1`, 50 decisions per TTI)
made bootstrapped Q-learning unstable with `γ = 1`. Twenty-five decisions per
TTI learned reliably. The greedy reference decides at the same granularity, so
the two are compared like for like.

* **Observation** (`4N + 4` values):
  * **Per UE:** its per-RB rate (normalised), its share of the TTI's RBs so
    far, its throughput relative to the fair share `T₀ / N`, and whether it
    has been served yet.
  * **Global:** the current throughput term `min(T/T₀, 1)`, the current Jain
    index, the share of UEs not yet served, and the fraction of decisions left.
    These are the ingredients of `R`, so the network can work out what one
    more RB group to a given UE would do to it.
* **Reward**: the change in `R` caused by the decision. The rewards of one TTI
  sum to `R(final) − R(empty)`. `R(empty) = −λ` is a constant, so with
  `γ = 1` maximising the return is the same as maximising the TTI objective,
  and nothing is lost by decomposing it.
* **The executed action is the stored action.** This was the bug that made
  the v1 agent unable to learn (see [HISTORY.md](HISTORY.md)).

### 4.2 Network ([`rl/dqn.py`](../src/drl5g/rl/dqn.py))

The Q-network is **permutation-equivariant**. A single MLP (`16 → 128 → 128 → 1`)
scores every UE from:

* that UE's own four features,
* the mean and max of each feature over all UEs (context), and
* the four global features.

Relabelling the UEs relabels the Q-values in the same way, so the network
does not have to learn every ordering of the same cell. It also has the same
weights for any `N`, so a model trained on one scenario runs on another. In a
pilot on scenario A, a conventional MLP over the flat observation needed four
times as many episodes and still scored lower.

### 4.3 Learning ([`rl/trainer.py`](../src/drl5g/rl/trainer.py))

Double DQN (the online network selects the next action, the target network
evaluates it), Huber loss, Adam (lr 5·10⁻⁴, annealed linearly to 5 % of that
over training), gradient-norm clip 10, replay
buffer of 10⁵ transitions, batch 64, hard target update every 1000 steps, and
ε decaying linearly from 1.0 to 0.02 over the first 60 % of training. One
training episode is one TTI on a freshly dropped cell.

The learning-rate annealing matters because near the optimum, many UEs have
almost the same Q-value. In a pilot without it, a single late update flipped
the greedy policy of scenario D, and validation dropped from 0.49 to 0.07.

Every 100 episodes the greedy policy is scored on 128 fixed validation drops.
The checkpoint keeps the best-scoring weights. Training, validation and
evaluation use independent random streams, so no evaluation drop is seen
during training. Training is deterministic for a given seed (this is
tested).

## 5. Evaluation protocol ([`evaluation.py`](../src/drl5g/evaluation.py))

* **5 seeds × 100 episodes = 500 independent UE drops per scenario**, each run
  for 200 TTIs.
* **Common random numbers.** For a given seed every scheduler gets an
  identically seeded environment, so all schedulers face the same 500 drops
  and differences between them are paired.
* **Reported per scheduler:** mean objective `R`, cell throughput, per-TTI
  Jain index, long-term Jain index (over episode-average throughputs), the
  worst-served UE's throughput, and the share of UEs that got nothing for the
  whole episode. Each comes with a 95 % confidence half-width (`1.96·s/√n`
  over drops).
* **DQN vs. the best other scheduler:** paired difference over the same drops,
  with a 95 % CI (printed by `drl5g evaluate`).

## 6. Limitations

* **Static, flat channel.** There is no fading, so there is no multi-user
  diversity for proportional fair or the agent to exploit. Adding
  per-RB/per-TTI fading would make the problem time-varying and is the most
  valuable extension.
* **Full-buffer traffic.** UEs always have data. There are no queues,
  arrivals or deadlines.
* **Single cell.** There is no interference, handover or mobility within an
  episode.
* **Calibrated link budget.** It is not a 3GPP model: the transmit power is
  applied per RB with a calibrated intercept, there is no noise figure, and
  Shannon capacity is not capped by an MCS table.
* **One training seed per scenario.** Run-to-run variance of training is not
  characterised. The evaluation CIs cover drop variance only.
