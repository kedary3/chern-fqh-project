"""Diagonalization helpers for closed-system Chern-number calculations."""

from __future__ import annotations

import numpy as np
import scipy.linalg as la

from .hofstadter import single_particle_hofstadter


def low_energy_dense(H: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return the lowest k eigenvalues and eigenvectors of a dense Hermitian matrix."""
    evals, evecs = la.eigh(H)
    return evals[:k], evecs[:, :k].T


def single_particle_twist_grid(
    Lx: int,
    Ly: int,
    alpha: float,
    grid: int,
    band_index: int = 0,
    t: float = 1.0,
) -> np.ndarray:
    """Return eigenvectors for one single-particle band over a twist grid."""
    dim = Lx * Ly
    states = np.zeros((grid, grid, dim), dtype=np.complex128)
    for ix, theta_x in enumerate(np.linspace(0.0, 2.0 * np.pi, grid, endpoint=False)):
        for iy, theta_y in enumerate(np.linspace(0.0, 2.0 * np.pi, grid, endpoint=False)):
            H = single_particle_hofstadter(Lx, Ly, alpha, theta_x, theta_y, t=t)
            evals, evecs = la.eigh(H)
            states[ix, iy] = evecs[:, band_index]
    return states


def quspin_many_body_hamiltonian(*args, **kwargs):
    """
    Placeholder for a QuSpin many-body Hamiltonian builder.

    QuSpin operator strings depend on the particle statistics and basis choice.
    This function is intentionally explicit about being a next implementation step
    rather than silently returning an incorrect Hamiltonian.
    """
    raise NotImplementedError(
        "Implement this with quspin.basis.spinless_fermion_basis_general or "
        "boson_basis_general, using hopping_terms() and density_density_terms()."
    )
