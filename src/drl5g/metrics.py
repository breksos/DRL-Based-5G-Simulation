"""Throughput, fairness and the scheduling objective shared by every component."""

from __future__ import annotations

import numpy as np


def jain_index(throughputs: np.ndarray) -> float:
    """Jain's fairness index in ``[1/n, 1]``; 1 means every user got the same rate.

    An all-zero vector is reported as 0: nobody being served is not "fair".
    """
    x = np.asarray(throughputs, dtype=np.float64)
    total = x.sum()
    if total <= 0.0:
        return 0.0
    return float(total**2 / (x.size * np.square(x).sum()))


def per_ue_throughput(allocation: np.ndarray, rb_capacity_bps: np.ndarray) -> np.ndarray:
    """Per-UE throughput (bit/s) for an allocation.

    Args:
        allocation: ``(n_rb,)`` array; entry ``i`` is the UE that gets RB ``i``.
        rb_capacity_bps: ``(n_ue,)`` capacity of one RB for each UE.
    """
    n_ue = rb_capacity_bps.shape[0]
    allocation = np.asarray(allocation, dtype=np.int64)
    if allocation.size and (allocation.min() < 0 or allocation.max() >= n_ue):
        raise ValueError(f"allocation contains UE indices outside [0, {n_ue})")
    rbs_per_ue = np.bincount(allocation, minlength=n_ue)
    return rbs_per_ue * rb_capacity_bps


def starved_fraction(throughputs: np.ndarray) -> float:
    """Fraction of UEs that received nothing."""
    return float(np.mean(np.asarray(throughputs) <= 0.0))


def scheduling_objective(
    throughputs_bps: np.ndarray,
    alpha: float,
    beta: float,
    throughput_norm_mbps: float,
    starvation_penalty: float = 0.0,
) -> float:
    """The per-TTI reward every scheduler is judged on.

    ``R = α · min(Σ throughput / norm, 1) + β · Jain(throughput) − λ · starved``

    The throughput term saturates at ``throughput_norm_mbps``: past that point
    only fairness can still raise ``R``, which makes the problem a real
    throughput/fairness trade-off rather than "serve the best UE".

    Jain's index alone barely penalises leaving a UE with nothing, so an
    optimiser of the first two terms learns to drop cell-edge users. The
    starvation term ``λ`` (the share of UEs served nothing this TTI) is a
    minimum-service QoS requirement that rules this out; ``λ = 0`` gives the
    original course-project objective.
    """
    total_mbps = float(np.sum(throughputs_bps)) / 1e6
    throughput_term = min(total_mbps / throughput_norm_mbps, 1.0)
    reward = alpha * throughput_term + beta * jain_index(throughputs_bps)
    if starvation_penalty:
        reward -= starvation_penalty * starved_fraction(throughputs_bps)
    return reward
