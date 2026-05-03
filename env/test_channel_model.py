# -*- coding: utf-8 -*-
"""Unit tests for channel_model.py -- run with: python env/test_channel_model.py"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from env.channel_model import snr_db, capacity_bps, ue_channel, random_ue_distances

def test_snr_range():
    s_100 = snr_db(100.0)
    s_500 = snr_db(500.0)
    assert 20 <= s_100 <= 25, f"100m SNR expected 20-25 dB, got {s_100:.2f}"
    assert 0 <= s_500 <= 5,   f"500m SNR expected 0-5 dB, got {s_500:.2f}"
    print(f"  SNR@100m = {s_100:.2f} dB  OK")
    print(f"  SNR@500m = {s_500:.2f} dB  OK")

def test_no_negative_snr():
    distances = np.linspace(50, 2000, 500)
    snrs = snr_db(distances)
    assert np.all(snrs >= 0), "Negative SNR found -- clamp broken"
    print(f"  No negative SNR over 50-2000m range  OK")

def test_monotone_decrease():
    distances = np.array([50., 100., 200., 300., 400., 500.])
    snrs = snr_db(distances)
    for i in range(len(snrs) - 1):
        assert snrs[i] >= snrs[i+1], f"SNR not monotone at d={distances[i]}m"
    print(f"  SNR decreases monotonically with distance  OK")

def test_capacity_positive():
    snrs = np.array([0.0, 5.0, 10.0, 20.0])
    caps = capacity_bps(snrs)
    assert np.all(caps > 0), "Capacity must be positive"
    print(f"  Capacities (Mbps): {caps/1e6}")
    print(f"  All positive  OK")

def test_array_input():
    distances = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    s, c = ue_channel(distances)
    assert s.shape == (5,) and c.shape == (5,)
    print(f"  Array input works, shapes {s.shape}  OK")

def test_random_distances():
    rng = np.random.default_rng(42)
    d = random_ue_distances(10, rng=rng)
    assert d.shape == (10,)
    assert np.all(d >= 50) and np.all(d <= 500)
    print(f"  Random distances all in [50, 500] m  OK")

if __name__ == '__main__':
    tests = [test_snr_range, test_no_negative_snr, test_monotone_decrease,
             test_capacity_positive, test_array_input, test_random_distances]
    passed = 0
    for t in tests:
        print(f"\n{t.__name__}")
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
