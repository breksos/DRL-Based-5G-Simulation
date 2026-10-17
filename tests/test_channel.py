import numpy as np
import pytest

from drl5g.channel import MIN_DISTANCE_M, capacity_bps, drop_ues, snr_db


def test_snr_matches_calibration_points():
    assert snr_db(100.0) == pytest.approx(22.0, abs=0.1)
    assert snr_db(500.0) == pytest.approx(-4.3, abs=0.1)


def test_snr_decreases_with_distance():
    snr = snr_db(np.linspace(MIN_DISTANCE_M, 1000.0, 200))
    assert np.all(np.diff(snr) < 0)


def test_capacity_is_positive_and_increasing():
    caps = capacity_bps(np.array([-10.0, 0.0, 10.0, 30.0]))
    assert np.all(caps > 0)
    assert np.all(np.diff(caps) > 0)
    assert capacity_bps(0.0) == pytest.approx(180e3)  # log2(1 + 1) = 1 bit/s/Hz


def test_drop_ues_respects_cell_and_is_reproducible():
    d1, a1 = drop_ues(1000, 500.0, np.random.default_rng(0))
    d2, a2 = drop_ues(1000, 500.0, np.random.default_rng(0))
    np.testing.assert_array_equal(d1, d2)
    np.testing.assert_array_equal(a1, a2)
    assert d1.min() >= MIN_DISTANCE_M and d1.max() <= 500.0
    assert a1.min() >= 0.0 and a1.max() < 2 * np.pi


def test_drop_ues_is_uniform_over_area():
    distances, _ = drop_ues(20_000, 500.0, np.random.default_rng(0))
    # Area-uniform: the outer ring r > R/√2 holds about half of the (annulus) area.
    ring_area = (500**2 - (500 / np.sqrt(2)) ** 2) / (500**2 - MIN_DISTANCE_M**2)
    assert np.mean(distances > 500 / np.sqrt(2)) == pytest.approx(ring_area, abs=0.02)
