"""Small fixed-particle-number many-body builders for spinless fermions.

These routines provide a dense fallback and a transparent reference backend for
small FQHE/FCI inverse-method experiments.  The scripts prefer QuSpin when it is
available, but these functions keep the project runnable in minimal Python
environments and are also convenient for tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .hofstadter import hopping_terms
from .lattice import nearest_neighbor_pairs, site_index


@dataclass(frozen=True)
class FixedNBasis:
    """Fixed-N bitstring basis for spinless fermions."""

    n_sites: int
    n_particles: int
    states: tuple[int, ...]
    index: dict[int, int]


def fixed_n_basis(n_sites: int, n_particles: int) -> FixedNBasis:
    if not (0 <= n_particles <= n_sites):
        raise ValueError("Require 0 <= n_particles <= n_sites.")
    states = []
    for occ in combinations(range(n_sites), n_particles):
        bitstring = 0
        for i in occ:
            bitstring |= 1 << i
        states.append(bitstring)
    states_tuple = tuple(states)
    return FixedNBasis(n_sites, n_particles, states_tuple, {s: k for k, s in enumerate(states_tuple)})


def _popcount(x: int) -> int:
    return int(x.bit_count())


def _annihilate(state: int, site: int) -> tuple[int, int] | None:
    if ((state >> site) & 1) == 0:
        return None
    sign = -1 if (_popcount(state & ((1 << site) - 1)) % 2) else 1
    return sign, state & ~(1 << site)


def _create(state: int, site: int) -> tuple[int, int] | None:
    if ((state >> site) & 1) == 1:
        return None
    sign = -1 if (_popcount(state & ((1 << site) - 1)) % 2) else 1
    return sign, state | (1 << site)


def cdag_c_matrix(basis: FixedNBasis, i: int, j: int) -> np.ndarray:
    """Matrix for c_i^dagger c_j in the fixed-N basis."""
    dim = len(basis.states)
    mat = np.zeros((dim, dim), dtype=np.complex128)
    for col, state in enumerate(basis.states):
        step1 = _annihilate(state, j)
        if step1 is None:
            continue
        sign1, state1 = step1
        step2 = _create(state1, i)
        if step2 is None:
            continue
        sign2, state2 = step2
        row = basis.index[state2]
        mat[row, col] += sign1 * sign2
    return mat


def number_matrix(basis: FixedNBasis, i: int) -> np.ndarray:
    """Matrix for n_i."""
    diag = [((state >> i) & 1) for state in basis.states]
    return np.diag(diag).astype(np.complex128)


def density_density_matrix(basis: FixedNBasis, i: int, j: int) -> np.ndarray:
    """Matrix for n_i n_j."""
    diag = [((state >> i) & 1) * ((state >> j) & 1) for state in basis.states]
    return np.diag(diag).astype(np.complex128)


def hermitian_hopping_matrix(basis: FixedNBasis, i: int, j: int, phase: complex = 1.0) -> np.ndarray:
    """
    Matrix for phase*c_i^dagger*c_j + conj(phase)*c_j^dagger*c_i.
    """
    mat = phase * cdag_c_matrix(basis, i, j) + np.conj(phase) * cdag_c_matrix(basis, j, i)
    return 0.5 * (mat + mat.conj().T)


def hofstadter_many_body_hamiltonian_dense(
    Lx: int,
    Ly: int,
    n_particles: int,
    alpha: float,
    *,
    theta_x: float = 0.0,
    theta_y: float = 0.0,
    t: float = 1.0,
    V: float = 0.0,
) -> tuple[np.ndarray, FixedNBasis]:
    """Dense interacting Harper-Hofstadter Hamiltonian in a fixed-N sector."""
    basis = fixed_n_basis(Lx * Ly, n_particles)
    dim = len(basis.states)
    H = np.zeros((dim, dim), dtype=np.complex128)

    # hopping_terms already includes both Hermitian-conjugate orientations.
    for i, j, amp in hopping_terms(Lx, Ly, alpha, theta_x, theta_y, t=t):
        H += amp * cdag_c_matrix(basis, i, j)

    for i, j in nearest_neighbor_pairs(Lx, Ly):
        H += V * density_density_matrix(basis, i, j)

    return 0.5 * (H + H.conj().T), basis


def grouped_fci_operator_library(
    Lx: int,
    Ly: int,
    n_particles: int,
    alpha: float,
    *,
    theta_x: float = 0.0,
    theta_y: float = 0.0,
    include_onsite: bool = True,
):
    """
    Physically organized Hermitian operator library for FCI inverse discovery.

    The first operators are the grouped Hofstadter kinetic term and the grouped
    nearest-neighbor density repulsion.  Optional single-site density operators
    let EHC test whether the target requires disorder or chemical-potential
    structure.
    """
    from .inverse_ehc import OperatorTerm

    basis = fixed_n_basis(Lx * Ly, n_particles)
    dim = len(basis.states)
    kinetic = np.zeros((dim, dim), dtype=np.complex128)
    for i, j, amp in hopping_terms(Lx, Ly, alpha, theta_x, theta_y, t=1.0):
        kinetic += amp * cdag_c_matrix(basis, i, j)
    kinetic = 0.5 * (kinetic + kinetic.conj().T)

    nn = np.zeros_like(kinetic)
    for i, j in nearest_neighbor_pairs(Lx, Ly):
        nn += density_density_matrix(basis, i, j)

    operators = [
        OperatorTerm("Hofstadter kinetic K(alpha,theta)", kinetic),
        OperatorTerm("nearest-neighbor repulsion sum_<ij> n_i n_j", nn),
    ]

    if include_onsite:
        for y in range(Ly):
            for x in range(Lx):
                site = site_index(x, y, Lx, Ly)
                operators.append(OperatorTerm(f"onsite density n_({x},{y})", number_matrix(basis, site)))

    return operators, basis


def expanded_fci_operator_library(
    Lx: int,
    Ly: int,
    n_particles: int,
    alpha: float,
    *,
    theta_x: float = 0.0,
    theta_y: float = 0.0,
):
    """
    Less biased local operator library with individual bonds and interactions.

    This is useful for Hamiltonian discovery because EHC can decide which local
    terms are needed, rather than being given the grouped Hofstadter Hamiltonian
    as a single basis vector.
    """
    from .inverse_ehc import OperatorTerm

    basis = fixed_n_basis(Lx * Ly, n_particles)
    operators: list[OperatorTerm] = []

    seen_bonds: set[tuple[int, int]] = set()
    for x in range(Lx):
        for y in range(Ly):
            i = site_index(x, y, Lx, Ly)

            xp = (x + 1) % Lx
            jx = site_index(xp, y, Lx, Ly)
            bx = tuple(sorted((i, jx)))
            if bx not in seen_bonds:
                seen_bonds.add(bx)
                phase_x = np.exp(1j * theta_x / Lx)
                operators.append(
                    OperatorTerm(
                        f"x-bond ({x},{y})->({xp},{y})",
                        hermitian_hopping_matrix(basis, jx, i, -phase_x),
                    )
                )

            yp = (y + 1) % Ly
            jy = site_index(x, yp, Lx, Ly)
            by = tuple(sorted((i, jy)))
            if by not in seen_bonds:
                seen_bonds.add(by)
                phase_y = np.exp(1j * (2.0 * np.pi * alpha * x + theta_y / Ly))
                operators.append(
                    OperatorTerm(
                        f"y-bond ({x},{y})->({x},{yp})",
                        hermitian_hopping_matrix(basis, jy, i, -phase_y),
                    )
                )

    for i, j in nearest_neighbor_pairs(Lx, Ly):
        operators.append(OperatorTerm(f"density repulsion n_{i} n_{j}", density_density_matrix(basis, i, j)))

    return operators, basis


def quspin_grouped_fci_operator_library(
    Lx: int,
    Ly: int,
    n_particles: int,
    alpha: float,
    *,
    theta_x: float = 0.0,
    theta_y: float = 0.0,
    include_onsite: bool = True,
):
    """
    QuSpin-backed grouped FCI operator library.

    This is the preferred backend for production exact diagonalization because
    QuSpin handles the many-body basis and operator construction.  The dense
    matrices returned here are only for the small inverse-method QCM analysis.
    """
    from quspin.basis import spinless_fermion_basis_general
    from quspin.operators import hamiltonian

    from .inverse_ehc import OperatorTerm

    n_sites = Lx * Ly
    basis = spinless_fermion_basis_general(n_sites, Nf=n_particles)

    def as_dense(static):
        H = hamiltonian(
            static,
            [],
            basis=basis,
            dtype=np.complex128,
            check_herm=False,
            check_symm=False,
            check_pcon=False,
        )
        return H.toarray()

    hop_couplings = [[amp, i, j] for i, j, amp in hopping_terms(Lx, Ly, alpha, theta_x, theta_y, t=1.0)]
    nn_couplings = [[1.0, i, j] for i, j in nearest_neighbor_pairs(Lx, Ly)]

    operators = [
        OperatorTerm("QuSpin Hofstadter kinetic K(alpha,theta)", as_dense([["+-", hop_couplings]])),
        OperatorTerm("QuSpin nearest-neighbor repulsion sum_<ij> n_i n_j", as_dense([["nn", nn_couplings]])),
    ]

    if include_onsite:
        for y in range(Ly):
            for x in range(Lx):
                site = site_index(x, y, Lx, Ly)
                operators.append(OperatorTerm(f"QuSpin onsite density n_({x},{y})", as_dense([["n", [[1.0, site]]]])))

    return operators, basis


def quspin_hofstadter_many_body_hamiltonian_dense(
    Lx: int,
    Ly: int,
    n_particles: int,
    alpha: float,
    *,
    theta_x: float = 0.0,
    theta_y: float = 0.0,
    t: float = 1.0,
    V: float = 0.0,
):
    """Build the target interacting Harper-Hofstadter Hamiltonian with QuSpin."""
    operators, basis = quspin_grouped_fci_operator_library(
        Lx,
        Ly,
        n_particles,
        alpha,
        theta_x=theta_x,
        theta_y=theta_y,
        include_onsite=False,
    )
    H = t * operators[0].matrix + V * operators[1].matrix
    return 0.5 * (H + H.conj().T), basis
