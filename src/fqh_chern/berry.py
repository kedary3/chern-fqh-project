"""Gauge-invariant Berry curvature and Chern number utilities."""

from __future__ import annotations

import numpy as np


def phase(z: complex, eps: float = 1e-14) -> complex:
    """Return z / |z| with a guard against singular overlaps."""
    abs_z = abs(z)
    if abs_z < eps:
        raise ValueError("Overlap magnitude is too small; Berry link is ill-conditioned.")
    return z / abs_z


def abelian_link(psi_a: np.ndarray, psi_b: np.ndarray) -> complex:
    """Compute a normalized U(1) link variable between two normalized states."""
    return phase(np.vdot(psi_a, psi_b))


def abelian_chern(states: np.ndarray) -> float:
    """
    Compute the Abelian Chern number from states[ix, iy, dim].

    Periodic boundary conditions in twist space are assumed.
    """
    Nx, Ny, _ = states.shape
    total_flux = 0.0

    for ix in range(Nx):
        for iy in range(Ny):
            psi = states[ix, iy]
            psi_x = states[(ix + 1) % Nx, iy]
            psi_y = states[ix, (iy + 1) % Ny]
            psi_xy = states[(ix + 1) % Nx, (iy + 1) % Ny]

            Ux = abelian_link(psi, psi_x)
            Uy_x = abelian_link(psi_x, psi_xy)
            Ux_y = abelian_link(psi_y, psi_xy)
            Uy = abelian_link(psi, psi_y)

            plaquette = Ux * Uy_x / (Ux_y * Uy)
            total_flux += np.angle(plaquette)

    return float(total_flux / (2.0 * np.pi))


def nonabelian_link(subspace_a: np.ndarray, subspace_b: np.ndarray) -> complex:
    """
    Compute a normalized non-Abelian link from two subspace frames.

    subspace_a and subspace_b have shape (q, dim), where q is the dimension of
    the ground-state multiplet.
    """
    overlap = subspace_a.conj() @ subspace_b.T
    return phase(np.linalg.det(overlap))


def nonabelian_chern(subspaces: np.ndarray) -> float:
    """
    Compute the non-Abelian Chern number from subspaces[ix, iy, q, dim].

    This is the robust diagnostic for a nearly degenerate ground-state manifold.
    """
    Nx, Ny, _, _ = subspaces.shape
    total_flux = 0.0

    for ix in range(Nx):
        for iy in range(Ny):
            P = subspaces[ix, iy]
            Px = subspaces[(ix + 1) % Nx, iy]
            Py = subspaces[ix, (iy + 1) % Ny]
            Pxy = subspaces[(ix + 1) % Nx, (iy + 1) % Ny]

            Ux = nonabelian_link(P, Px)
            Uy_x = nonabelian_link(Px, Pxy)
            Ux_y = nonabelian_link(Py, Pxy)
            Uy = nonabelian_link(P, Py)

            plaquette = Ux * Uy_x / (Ux_y * Uy)
            total_flux += np.angle(plaquette)

    return float(total_flux / (2.0 * np.pi))
