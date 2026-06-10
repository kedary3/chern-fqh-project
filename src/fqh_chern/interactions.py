"""Interaction terms for lattice FQH-like models."""

from __future__ import annotations

from .lattice import nearest_neighbor_pairs


def density_density_terms(Lx: int, Ly: int, V: float) -> list[tuple[int, int, float]]:
    """Return nearest-neighbor density-density interaction terms V n_i n_j."""
    return [(i, j, float(V)) for i, j in nearest_neighbor_pairs(Lx, Ly)]
