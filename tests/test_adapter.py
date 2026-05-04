import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.train import ActionAdapter


def test_action_adapter_uses_state():
    adapter = ActionAdapter(n_ue=10, n_rb=50)
    alloc0 = adapter.to_allocation(0)
    alloc1 = adapter.to_allocation(1)
    # All RBs must change when the chosen UE changes — no hardcoded round-robin half.
    assert np.all(alloc0 == 0), "All RBs should go to UE 0"
    assert np.all(alloc1 == 1), "All RBs should go to UE 1"
    assert np.sum(alloc0 != alloc1) == 50, "100% of RBs should differ for different UE choices"
