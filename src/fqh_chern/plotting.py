"""Plotting helpers."""

from __future__ import annotations

import numpy as np


def plot_spectrum(evals: np.ndarray, ax=None):
    """Plot eigenvalues indexed by level number."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(np.arange(len(evals)), np.sort(np.real(evals)), marker="o", linestyle="none")
    ax.set_xlabel("Level index")
    ax.set_ylabel("Energy")
    return ax
