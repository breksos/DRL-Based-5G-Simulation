# Project Summary — What You Need to Know for the Presentation

**Project:** Deep Reinforcement Learning Based Resource Allocation in a Simplified 5G Downlink Network  
**Your role:** 5G Environment Simulation (the "world" the AI learns in)

---

## The Big Picture — One Sentence

> We built a simplified 5G cell simulator and trained a Deep Q-Network to decide which user gets which radio resource, balancing total network speed against fairness.

---

## The Problem We Are Solving

A 5G base station (gNodeB) has a fixed pool of radio resources called **Resource Blocks (RBs)**. Every millisecond it must decide: *which user gets which RB?*

- If you give all RBs to the strongest user → great speed, but weak users get nothing (unfair)
- If you split equally → fair, but you waste resources on users with bad channels
- **We want the AI to find the best middle ground automatically**

This decision problem is called **Radio Resource Management (RRM)** and it happens millions of times per second in real networks.

---

## The 5G Concepts You Need to Know

### gNodeB (gNB)
The 5G base station. One per cell. In our model there is exactly **1 gNB at the center**. It serves all users.

### UE (User Equipment)
Any user device — phone, laptop. In our model: **10–20 UEs** placed randomly within the cell, each at a different distance from the gNB.

### Resource Block (RB)
The smallest chunk of radio spectrum the gNB can assign. Think of it as one "slot" of bandwidth.  
- Each RB has a bandwidth of **180 kHz** (12 subcarriers × 15 kHz)
- Our cell has **50–100 RBs** total per time step
- Each step the scheduler assigns every RB to exactly one UE

### Path Loss
Signal gets weaker with distance. We model it as:
```
PL(d) = 70.25 + 37.6 × log₁₀(d)     [d in metres]
```
At 100 m → SNR ≈ 22 dB (strong)  
At 500 m → SNR ≈ 0 dB (barely anything)

### SNR (Signal-to-Noise Ratio)
How strong a user's signal is relative to noise. Higher = better channel.  
**This is the core input to all scheduling decisions.**

### Shannon Capacity
The maximum data rate a channel can carry:
```
C = B × log₂(1 + SNR_linear)     [bps per RB]
```
At SNR = 22 dB → ~1.2 Mbps per RB  
At SNR = 0 dB  → ~0.18 Mbps per RB  
**You cannot exceed this limit regardless of how many RBs you assign.**

### Jain's Fairness Index
Measures how evenly throughput is distributed across users. Always between 0 and 1.
```
J = (Σ xᵢ)² / (N × Σ xᵢ²)
```
- J = 1.0 → everyone gets exactly the same throughput (perfectly fair)
- J = 0.1 → one user gets everything, others get nothing (very unfair)
- J > 0.7 → generally considered acceptable

---

## How the Simulation Works

### State (what the AI sees each step)
A vector of 70 numbers (for Scenario A):
```
[ snr_1, snr_2, ..., snr_10,              ← 10 SNR values (dB)
  throughput_1, ..., throughput_10,        ← 10 throughput values (Mbps)
  rb_occupancy_1, ..., rb_occupancy_50 ]  ← 50 RB fill values
```

### Action (what the AI decides)
An array of 50 integers, one per RB:
```
action = [3, 3, 7, 1, 1, 1, 0, ...]   ← "RB0→UE3, RB1→UE3, RB2→UE7, ..."
```

### Reward (how we score the AI's decision)
```
R = α × total_throughput_Mbps + β × Jain_fairness
```
Default: α = β = 0.5 → we care equally about speed and fairness.

### Episode
One full simulation run = **200 steps**. With `mobility=False` (default), UE positions re-randomise each episode. With `mobility=True`, positions drift ±10 m per episode reset instead of fully re-randomising.

---

## The Three Baseline Schedulers (Emre's work)

These are the "dumb" algorithms the DQN must beat.

### Random
Assigns each RB to a random user. No intelligence.  
→ Throughput: ~29 Mbps, Fairness: ~0.58

### Round Robin (RR)
Cycles through users in order: RB0→UE0, RB1→UE1, ..., RB9→UE0, ...  
Every user gets the same number of RBs regardless of channel quality.  
→ Throughput: ~30 Mbps, Fairness: ~0.65  
→ **Fairest baseline, but wastes RBs on weak users**

### Proportional Fair (PF)
For each RB independently, picks the UE with the best `instant_rate / avg_rate`
ratio and EMA-updates that UE's avg_rate before the next RB. Prioritises users
who have good channel AND haven't received much recently.  
→ Throughput: ~57–116 Mbps (highest of any scheduler), Fairness: ~0.27–0.31  
→ **Throughput leader; the reward clamp at 50 Mbps is what neutralises that
advantage in the combined metric**

---

## The DQN Agent (İsmail's work)

### What is DQN?
Deep Q-Network = Q-Learning + Neural Network.

**Q-Learning:** Learn the value Q(state, action) = "if I take this action now and play perfectly forever, how much total reward do I get?"

**Why neural network?** Our state has 70 dimensions. A lookup table would need 10^70 entries. A neural network approximates the Q-function instead.

### Key components
- **Q-Network:** 2 hidden layers × 256 neurons → outputs Q-value for each possible action
- **Experience Replay:** Stores past (state, action, reward, next_state) tuples in a buffer of 100,000. Trains on random mini-batches to break correlation between consecutive experiences
- **Target Network:** A frozen copy of Q-Network updated every 1000 steps. Prevents the training target from moving too fast (stabilises learning)
- **ε-greedy:** Starts exploring randomly (ε=1.0), gradually shifts to exploiting learned policy (ε→0.01 over 50,000 steps)

### Training loop
```
for each episode:
    state = env.reset()
    while not done:
        action = agent.select_action(state)   # ε-greedy
        next_state, reward, done = env.step(action)
        agent.store(state, action, reward, next_state, done)
        agent.train_step()                    # sample buffer, update Q-network
        state = next_state
```

---

## The Four Scenarios

| Scenario | UEs | RBs | RBs per UE | What it tests |
|---|---|---|---|---|
| A | 10 | 50 | 5 | Baseline — balanced |
| B | 15 | 75 | 5 | More users, same ratio |
| C | 20 | 100 | 5 | Dense network |
| D | 20 | 50 | 2.5 | **Resource-scarce** — hardest scheduling problem |

Scenario D is the most interesting: there are not enough RBs for everyone. The trade-off between throughput and fairness is sharpest here.

---

## Your Part — What You Built

### `env/channel_model.py`
Functions that model the radio channel:
- `path_loss_db(d)` — how much signal is lost at distance d
- `snr_db(d)` — the resulting SNR a UE experiences
- `capacity_bps(snr_db)` — Shannon capacity per RB

### `env/network_env.py`
The `FiveGEnv` class — a Gymnasium-compatible environment:
- `reset()` → randomly place UEs, compute their SNRs, return state vector
- `step(action)` → apply RB allocation, compute throughputs, compute reward, return next state
- Validates with `gymnasium.utils.env_checker` (the standard test for RL environments)
- Optional **mobility** mode: UEs drift ±10m per episode

### `gui/app.py`
Real-time dashboard with 5 live plots — lets you visually see what the scheduler is doing at every step. Run with `python main.py --gui`.

---

## What the GUI Shows — Quick Reference

| Plot | What it shows | What good looks like |
|---|---|---|
| Network Topology | UE positions, coloured by SNR (red=weak, green=strong) | Green dots near center, red near edge |
| SNR per UE | Bar chart of each UE's channel quality | Wide spread = interesting scheduling problem |
| Throughput per UE | How many Mbps each UE got this step | DQN: multiple bars non-zero, biased to high-SNR UEs |
| RB Allocation | Which UE got each RB (colour coded) | DQN: multi-colour clusters; PF: solid single colour |
| Live History | Reward and fairness over last 200 steps | DQN: rising curve during training, then stable plateau |

---

## Actual Results — Combined R per Scenario

Combined R = 0.5 × min(throughput / 50 Mbps, 1.0) + 0.5 × Jain_fairness  
Reported as mean over 5 seeds × 50 episodes per cell. See `results/baselines/comparison.csv`.

| Scenario | Random R | RR R | PF R | DQN R | Best |
|---|---|---|---|---|---|
| a (10UE/50RB)   | 0.5772 | 0.6074 | 0.6539 | 0.5764 | **PF** |
| b (15UE/75RB)   | 0.7133 | 0.7497 | 0.6520 | 0.6645 | **RR** |
| c (20UE/100RB)  | 0.7720 | 0.8111 | 0.6486 | 0.6328 | **RR** |
| d (20UE/50RB)   | 0.5299 | 0.5898 | 0.6334 | 0.4190 | **PF** |

Raw mean throughput (Mbps) — pre-clamp, so PF's throughput lead is visible:

| Scenario | Random | RR | PF | DQN |
|---|---|---|---|---|
| a | 28.58 | 28.58 |  56.75 | 29.21 |
| b | 43.41 | 43.42 |  86.45 | 44.52 |
| c | 57.63 | 57.62 | 116.03 | 59.01 |
| d | 28.81 | 28.81 |  60.81 | 29.58 |

**The DQN does not beat any classical baseline on combined R in any of the four scenarios.** The earlier "DQN beats PF in 3/4 scenarios" headline was a metric artifact: (i) PF was implemented as a strawman that gave all RBs to a single argmax UE per step, dragging its fairness to ~0.10; (ii) the reward clamps throughput at 50 Mbps, which removes PF's true advantage. With per-RB PF (the standard formulation) PF's fairness rises to ~0.27–0.31 and PF wins the combined metric in scenarios A and D outright. RR wins B and C. DQN ranks last in A, C, D and second-to-last in B.

---

## Likely Presentation Questions

**"Why doesn't your DQN beat the baselines?"**
> Three reasons, and we report this as a negative result. (a) **Action-space / Bellman-target mismatch.** The DQN outputs one Q-value per UE; the integer stored in the replay buffer is *not* the per-RB allocation that actually generated the reward. The reward is produced by the action adapter (softmax-multinomial), so the Q-target is regressed onto the wrong action. (b) **Stochastic adapter cannot match RR's fairness.** Even with a perfect Q-function, the per-RB softmax-multinomial adapter samples each RB independently — it cannot deterministically rotate UEs the way Round Robin does, so it cannot reach RR-level fairness. (c) **Compute / data limits.** Per-scenario training, 2000 episodes on CPU, single training seed. The architectural fix is a per-(UE, RB) Q-network whose head outputs `n_rb × n_ue` scores and assigns each RB to its argmax UE — the action stored in the buffer would then be the action that produced the reward. That refactor was out of scope here.

**"Why not just use Proportional Fair? It's already used in real 4G/5G."**
> Under our combined metric (α = β = 0.5) with throughput clamped at 50 Mbps, PF actually does win in the resource-scarce scenarios (A and D) — its raw throughput (~57–116 Mbps) is far above any other scheduler. RR wins B and C because its higher fairness compensates once PF's throughput is clipped. There is no scenario in our results where the DQN improves over both PF and RR; the project is a negative result on the methods side.

**"Why DQN and not a simpler algorithm?"**
> The state space is continuous with 70 dimensions. Tabular Q-learning would need a table with 10^70+ entries — impossible. DQN uses a neural network to approximate the Q-function across this continuous space.

**"What is Experience Replay for?"**
> Without it, training on consecutive steps is like studying the same page repeatedly — the network overfits to recent experience. Randomly sampling from a large buffer breaks this correlation and gives stable, generalisable learning.

**"Why does the DQN need a Target Network?"**
> Without it, you're chasing a moving target: the network updates both the prediction AND the target it's trying to hit simultaneously. The target network is frozen for 1000 steps, giving the training a stable goal.

**"What does the reward function α=β=0.5 mean?"**
> We treat throughput and fairness as equally important. With α=β=0.5, the DQN must learn to balance both rather than simply maximizing throughput.

**"Your model is too simple — no interference, single cell, no frequency-selective fading."**
> Acknowledged. This is a proof-of-concept. The MDP formulation, DQN architecture, and reward design are all transferable to more complex simulators (ns-3, simuLTE). We demonstrate the approach works; scaling is future work.

---

## Key Numbers to Quote

| Thing | Value |
|---|---|
| Cell radius | 500 m |
| UE distance range | 50–500 m |
| SNR at 100 m | 22 dB |
| SNR at 500 m | 0 dB |
| Mean SNR across episodes | 8.47 dB |
| UEs below 5 dB (near edge) | 48.3% |
| UEs above 20 dB (excellent) | 14.6% |
| Default RBs | 50 |
| RB bandwidth | 180 kHz |
| Capacity per RB at 22 dB | ~1.2 Mbps |
| DQN hidden layers | 2 × 256 neurons |
| Replay buffer size | 100,000 |
| Training episodes | 2,000 |
| ε decay steps | 50,000 |
