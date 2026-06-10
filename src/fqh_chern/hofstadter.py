"""Harper-Hofstadter Hamiltonian builders."""

from __future__ import annotations

import numpy as np

from .lattice import site_index


def hopping_terms(
    Lx: int,
    Ly: int,
    alpha: float,
    theta_x: float,
    theta_y: float,
    t: float = 1.0,
) -> list[tuple[int, int, complex]]:
    """
    Return nearest-neighbor Hofstadter hopping terms on a torus.

    Each tuple is (i, j, amplitude), corresponding to amplitude * c_i^dagger c_j.
    The gauge convention is Landau gauge with Peierls phase on y-directed hopping.
    """
    terms: list[tuple[int, int, complex]] = []

    for x in range(Lx):
        for y in range(Ly):
            i = site_index(x, y, Lx, Ly)

            xp = (x + 1) % Lx
            jx = site_index(xp, y, Lx, Ly)
            phase_x = np.exp(1j * theta_x / Lx)
            terms.append((jx, i, -t * phase_x))
            terms.append((i, jx, -t * np.conj(phase_x)))

            yp = (y + 1) % Ly
            jy = site_index(x, yp, Lx, Ly)
            phase_y = np.exp(1j * (2.0 * np.pi * alpha * x + theta_y / Ly))
            terms.append((jy, i, -t * phase_y))
            terms.append((i, jy, -t * np.conj(phase_y)))

    return terms


def single_particle_hofstadter(
    Lx: int,
    Ly: int,
    alpha: float,
    theta_x: float = 0.0,
    theta_y: float = 0.0,
    t: float = 1.0,
) -> np.ndarray:
    """Build the dense single-particle Harper-Hofstadter matrix."""
    dim = Lx * Ly
    H = np.zeros((dim, dim), dtype=np.complex128)
    for i, j, amp in hopping_terms(Lx, Ly, alpha, theta_x, theta_y, t=t):
        H[i, j] += amp
    return H
