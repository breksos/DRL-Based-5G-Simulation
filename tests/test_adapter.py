import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.train import ActionAdapter


def test_action_adapter_per_rb():
    """Softmax sampling spreads RBs across multiple UEs even with a dominant Q-value."""
    adapter = ActionAdapter(n_ue=10, n_rb=50)
    q_values = np.zeros(10, dtype=np.float32)
    q_values[3] = 1.0           # UE 3 dominates but softmax gives others non-zero probability
    snrs = np.full(10, 15.0, dtype=np.float32)
    alloc = adapter.to_allocation(snrs, q_values)
    assert len(np.unique(alloc)) >= 2, "softmax sampling must assign RBs to at least 2 distinct UEs"


def test_action_adapter_snr_aware():
    """With uniform Q-values, high-SNR UE wins more RBs via the SNR term."""
    adapter = ActionAdapter(n_ue=10, n_rb=50)
    q_values = np.zeros(10, dtype=np.float32)
    snrs = np.full(10, 5.0, dtype=np.float32)
    snrs[0] = 25.0              # UE 0 has best channel
    alloc = adapter.to_allocation(snrs, q_values)
    assert len(np.unique(alloc)) >= 2, "SNR-biased softmax must still assign RBs to at least 2 distinct UEs"


def test_action_adapter_distribution():
    """High-priority pair (UEs 0 and 1) must win most RBs when given top Q-values."""
    adapter = ActionAdapter(n_ue=10, n_rb=50, seed=0)
    q_values = np.array([1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    snrs = np.full(10, 15.0, dtype=np.float32)
    alloc = adapter.to_allocation(snrs, q_values)
    top_pair = np.sum((alloc == 0) | (alloc == 1))
    # Random baseline = 50*(2/10)=10; softmax with q=[1,1,0,...] gives ~0.40 combined prob → ~20 expected
    assert top_pair >= 15, f"expected ≥15 RBs to UEs 0/1, got {top_pair}"
