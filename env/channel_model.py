"""
5G downlink channel model — simplified log-distance model calibrated for
50–500 m micro-cell deployment.

Formula: PL(d) = PL0 + 10·n·log10(d)  [d in metres]
PL0 is calibrated (not the 3GPP COST-231 macro constant) so that the model
produces physically meaningful SNR diversity over the 50–500 m UE range:
  SNR ≈ 20–25 dB at 100 m,  SNR ≈ 0–5 dB at 500 m.
"""

import numpy as np

# --- Constants ---
PL0_DB = 70.25          # calibrated reference loss at d=1 m (dB)
PATH_LOSS_EXP = 3.76    # urban macro path-loss exponent
P_TX_DBM = 46.0         # gNodeB transmit power (dBm)
RB_BANDWIDTH_HZ = 180e3 # 12 subcarriers × 15 kHz per subcarrier
THERMAL_NOISE_DBM_PER_HZ = -174.0


def path_loss_db(distance_m: float) -> float:
    """Log-distance path loss in dB. distance_m must be > 0."""
    return PL0_DB + 10.0 * PATH_LOSS_EXP * np.log10(np.maximum(distance_m, 1.0))


def noise_power_dbm(bandwidth_hz: float = RB_BANDWIDTH_HZ) -> float:
    return THERMAL_NOISE_DBM_PER_HZ + 10.0 * np.log10(bandwidth_hz)


def snr_db(distance_m: float | np.ndarray) -> float | np.ndarray:
    """SNR in dB for a given distance (scalar or array)."""
    pl = path_loss_db(distance_m)
    n0 = noise_power_dbm()
    return np.maximum(P_TX_DBM - pl - n0, 0.0)  # clamp negatives to 0


def capacity_bps(snr_db_val: float | np.ndarray,
                 bandwidth_hz: float = RB_BANDWIDTH_HZ) -> float | np.ndarray:
    """Shannon capacity in bps for a single RB."""
    snr_linear = 10.0 ** (snr_db_val / 10.0)
    return bandwidth_hz * np.log2(1.0 + snr_linear)


def ue_channel(distance_m: float | np.ndarray):
    """Return (snr_db, capacity_bps) for given distance(s)."""
    s = snr_db(distance_m)
    c = capacity_bps(s)
    return s, c


def random_ue_distances(n_ue: int, d_min: float = 50.0, d_max: float = 500.0,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Uniform random distances for n_ue users."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(d_min, d_max, size=n_ue)
