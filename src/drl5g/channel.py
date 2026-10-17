"""Single-cell downlink channel model.

Log-distance path loss with no shadowing or fast fading, so a UE's SNR depends
only on its distance from the gNB and stays constant for a whole episode::

    PL(d)  = PL0 + 10 · n · log10(d)              [dB, d in metres]
    SNR(d) = P_tx − PL(d) − N0(B)                 [dB]
    C(SNR) = B · log2(1 + 10^(SNR / 10))          [bit/s per resource block]

``PL0`` and ``P_tx`` are calibrated together rather than taken from a 3GPP link
budget: they give ≈33 dB at 50 m, ≈22 dB at 100 m and ≈ −4 dB at 500 m, which
spreads UEs across the useful range of the Shannon curve.
"""

from __future__ import annotations

import numpy as np

PL0_DB = 70.25
"""Calibrated path-loss intercept at 1 m (dB)."""

PATH_LOSS_EXPONENT = 3.76
"""Urban path-loss exponent ``n``."""

TX_POWER_DBM = 46.0
"""gNB transmit power applied to each resource block (dBm)."""

RB_BANDWIDTH_HZ = 180e3
"""One LTE/NR numerology-0 resource block: 12 subcarriers × 15 kHz."""

THERMAL_NOISE_DBM_PER_HZ = -174.0

MIN_DISTANCE_M = 50.0
"""UEs are never dropped closer than this to the gNB."""


def path_loss_db(distance_m: float | np.ndarray) -> np.ndarray:
    """Log-distance path loss in dB. Distances below 1 m are treated as 1 m."""
    distance = np.maximum(np.asarray(distance_m, dtype=np.float64), 1.0)
    return PL0_DB + 10.0 * PATH_LOSS_EXPONENT * np.log10(distance)


def noise_power_dbm(bandwidth_hz: float = RB_BANDWIDTH_HZ) -> float:
    """Thermal noise power over ``bandwidth_hz`` in dBm."""
    return THERMAL_NOISE_DBM_PER_HZ + 10.0 * float(np.log10(bandwidth_hz))


def snr_db(distance_m: float | np.ndarray) -> np.ndarray:
    """Per-RB SNR in dB at the given distance(s)."""
    return TX_POWER_DBM - path_loss_db(distance_m) - noise_power_dbm()


def capacity_bps(snr: float | np.ndarray, bandwidth_hz: float = RB_BANDWIDTH_HZ) -> np.ndarray:
    """Shannon capacity in bit/s of one resource block at ``snr`` dB."""
    snr_linear = 10.0 ** (np.asarray(snr, dtype=np.float64) / 10.0)
    return bandwidth_hz * np.log2(1.0 + snr_linear)


def drop_ues(
    n_ue: int, cell_radius_m: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Drop ``n_ue`` users uniformly over the cell area.

    Returns ``(distances_m, angles_rad)``. Sampling the radius as
    ``sqrt(U)`` makes the density uniform per unit area, so more users land
    near the cell edge than near the gNB, as in a real deployment.
    """
    r_min_sq = (MIN_DISTANCE_M / cell_radius_m) ** 2
    distances = cell_radius_m * np.sqrt(rng.uniform(r_min_sq, 1.0, size=n_ue))
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_ue)
    return distances, angles
