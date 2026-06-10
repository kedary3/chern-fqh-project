"""Observable utilities."""

from __future__ import annotations

import numpy as np


def many_body_gap(evals: np.ndarray, multiplet: int) -> float:
    """Return the gap between the ground-state multiplet and the next state."""
    if len(evals) <= multiplet:
        raise ValueError("Need at least multiplet + 1 eigenvalues to compute a gap.")
    return float(evals[multiplet] - evals[multiplet - 1])


def multiplet_width(evals: np.ndarray, multiplet: int) -> float:
    """Return the energy spread inside the low-energy multiplet."""
    if multiplet < 1 or len(evals) < multiplet:
        raise ValueError("Invalid multiplet size.")
    return float(evals[multiplet - 1] - evals[0])
