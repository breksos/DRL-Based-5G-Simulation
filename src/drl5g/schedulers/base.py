"""The interface every scheduler implements."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np


class Scheduler(Protocol):
    """Assigns every resource block of one TTI to a UE.

    ``allocate`` receives the rate (bit/s) each UE would get from a single RB in
    this TTI and returns an ``(n_rb,)`` integer array of UE indices.
    ``reset`` is called at the start of every episode, after the environment
    has dropped a new set of UEs.
    """

    name: str

    def allocate(self, rb_capacity_bps: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None: ...


class LastInputCache:
    """Remembers the allocation computed for the most recent capacity vector.

    Deterministic schedulers see the same capacities for a whole episode
    (the channel is static), so recomputing every TTI would be wasted work.
    """

    def __init__(self) -> None:
        self._key: np.ndarray | None = None
        self._value: np.ndarray | None = None

    def get_or_compute(
        self, rb_capacity_bps: np.ndarray, compute: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        if self._key is None or not np.array_equal(rb_capacity_bps, self._key):
            self._key = np.array(rb_capacity_bps, copy=True)
            self._value = compute(rb_capacity_bps)
        return self._value.copy()
