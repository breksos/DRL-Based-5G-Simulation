"""
Evaluation metrics shared across schedulers and DQN agent.
All functions operate on plain numpy arrays.
"""

import numpy as np
from env.channel_model import capacity_bps, RB_BANDWIDTH_HZ


def jains_fairness(throughputs: np.ndarray) -> float:
    """Jain's Fairness Index in [0, 1]. 1 = perfectly fair."""
    s = throughputs.sum()
    if s == 0.0:
        return 1.0
    return float((s ** 2) / (len(throughputs) * (throughputs ** 2).sum() + 1e-12))


def per_ue_throughput(action: np.ndarray, snrs: np.ndarray,
                      bw_hz: float = RB_BANDWIDTH_HZ) -> np.ndarray:
    """
    Compute per-UE throughput (bps) given an RB allocation.
    action : (N_RB,) int array — each element is UE index
    snrs   : (N_UE,) float array — SNR in dB per UE
    """
    n_ue = len(snrs)
    tp = np.zeros(n_ue, dtype=np.float64)
    caps = capacity_bps(snrs, bw_hz)
    for ue in action:
        tp[int(ue) % n_ue] += caps[int(ue) % n_ue]
    return tp


def system_throughput(throughputs: np.ndarray) -> float:
    """Sum of per-UE throughputs in bps."""
    return float(throughputs.sum())


def evaluate_scheduler(env, scheduler, n_episodes: int = 100) -> dict:
    """
    Run a scheduler on an environment for n_episodes.
    Returns dict with mean/std of throughput (Mbps) and fairness.
    """
    ep_tp, ep_fair = [], []
    for _ in range(n_episodes):
        state, _ = env.reset()
        scheduler.reset()
        done = False
        step_tp, step_fair = [], []
        while not done:
            action = scheduler.allocate(state)
            state, _, terminated, truncated, info = env.step(action)
            step_tp.append(info['throughput_total'] / 1e6)
            step_fair.append(info['fairness'])
            done = terminated or truncated
        ep_tp.append(np.mean(step_tp))
        ep_fair.append(np.mean(step_fair))
    return {
        "avg_throughput_mbps": float(np.mean(ep_tp)),
        "std_throughput_mbps": float(np.std(ep_tp)),
        "avg_fairness":        float(np.mean(ep_fair)),
        "std_fairness":        float(np.std(ep_fair)),
    }
