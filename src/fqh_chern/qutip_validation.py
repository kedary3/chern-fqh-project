"""Optional QuTiP validation utilities."""

from __future__ import annotations

import numpy as np


def to_qobj(matrix: np.ndarray):
    """Convert a dense matrix to a QuTiP Qobj if QuTiP is installed."""
    try:
        import qutip as qt
    except ImportError as exc:
        raise ImportError("QuTiP is required for this validation utility.") from exc
    return qt.Qobj(matrix)


def qutip_eigenenergies(matrix: np.ndarray):
    """Compute eigenenergies with QuTiP for small dense validation problems."""
    return to_qobj(matrix).eigenenergies()
