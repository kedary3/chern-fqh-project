"""Eigenstate-to-Hamiltonian Construction (EHC) utilities.

This module implements the quantum covariance matrix inverse method from
Chertkov and Clark, arXiv:1802.01590.  The implementation is deliberately
backend-light: operators are NumPy/SciPy matrices, so targets obtained from
QuSpin, QuTiP, or a custom exact-diagonalization routine can all be analyzed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

ArrayLikeOperator = np.ndarray | sp.spmatrix


@dataclass(frozen=True)
class OperatorTerm:
    """A named Hermitian operator used as a basis element in EHC."""

    name: str
    matrix: ArrayLikeOperator


@dataclass(frozen=True)
class EHCSolution:
    """One candidate Hamiltonian discovered by the EHC covariance analysis."""

    eigenvalue: float
    coefficients: np.ndarray
    variance: float
    energy: complex
    residual_norm: float


def normalize_state(psi: np.ndarray, *, atol: float = 1e-14) -> np.ndarray:
    """Return a normalized state vector."""
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    norm = la.norm(psi)
    if norm < atol:
        raise ValueError("Cannot normalize a state with near-zero norm.")
    return psi / norm


def _apply(op: ArrayLikeOperator, psi: np.ndarray) -> np.ndarray:
    return op @ psi


def expectation(psi: np.ndarray, op: ArrayLikeOperator) -> complex:
    """Compute <psi|op|psi> for a normalized or unnormalized vector."""
    psi = normalize_state(psi)
    return np.vdot(psi, _apply(op, psi))


def quantum_covariance_matrix(
    psi: np.ndarray,
    operators: list[OperatorTerm],
    *,
    symmetrize: bool = True,
) -> np.ndarray:
    """
    Compute the EHC quantum covariance matrix.

    C_ab = <h_a h_b> - <h_a><h_b>.

    For Hermitian h_a, the resulting matrix is Hermitian and positive
    semidefinite up to numerical noise.  A null vector gives real coupling
    constants for a Hamiltonian for which psi has zero energy variance.
    """
    if not operators:
        raise ValueError("At least one operator is required.")

    psi = normalize_state(psi)
    acted = np.vstack([_apply(term.matrix, psi) for term in operators])
    means = np.array([np.vdot(psi, acted[a]) for a in range(len(operators))])
    qcm = acted.conj() @ acted.T - np.outer(means.conj(), means)

    if symmetrize:
        qcm = 0.5 * (qcm + qcm.conj().T)

    return qcm


def ehc_spectrum(
    psi: np.ndarray,
    operators: list[OperatorTerm],
    *,
    real_couplings: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Diagonalize the QCM.

    Returns eigenvalues, eigenvectors, and the QCM.  Eigenvectors are columns;
    column k contains the couplings of candidate Hamiltonian k in the supplied
    operator basis.
    """
    qcm = quantum_covariance_matrix(psi, operators)
    analysis_matrix = qcm.real if real_couplings else qcm
    analysis_matrix = 0.5 * (analysis_matrix + analysis_matrix.conj().T)
    evals, evecs = la.eigh(analysis_matrix)
    evals = np.maximum(evals.real, 0.0)
    return evals, evecs, qcm


def build_hamiltonian(
    operators: list[OperatorTerm],
    coefficients: np.ndarray,
) -> np.ndarray:
    """Construct a dense Hamiltonian from real EHC coefficients."""
    if len(operators) != len(coefficients):
        raise ValueError("Number of coefficients must match operator count.")

    first = operators[0].matrix
    dim = first.shape[0]
    H = np.zeros((dim, dim), dtype=np.complex128)
    for coeff, term in zip(coefficients, operators, strict=True):
        mat = term.matrix.toarray() if sp.issparse(term.matrix) else np.asarray(term.matrix)
        H += coeff * mat
    return 0.5 * (H + H.conj().T)


def energy_variance(psi: np.ndarray, H: ArrayLikeOperator) -> tuple[complex, float, float]:
    """Return energy, variance, and residual norm of psi under H."""
    psi = normalize_state(psi)
    Hpsi = _apply(H, psi)
    energy = np.vdot(psi, Hpsi)
    residual = Hpsi - energy * psi
    variance = float(np.real(np.vdot(residual, residual)))
    return energy, max(variance, 0.0), float(la.norm(residual))


def discover_hamiltonians(
    psi: np.ndarray,
    operators: list[OperatorTerm],
    *,
    n_solutions: int = 5,
) -> list[EHCSolution]:
    """
    Return the lowest-variance Hamiltonians in the target operator space.

    If the lowest QCM eigenvalue is zero within numerical tolerance, the
    associated Hamiltonian has the target state as an exact eigenstate inside
    the chosen operator space.  Otherwise, the lowest vectors are approximate
    parent Hamiltonian candidates.
    """
    evals, evecs, _ = ehc_spectrum(psi, operators, real_couplings=True)
    out: list[EHCSolution] = []
    for k in range(min(n_solutions, len(evals))):
        coeffs = np.asarray(evecs[:, k], dtype=float)
        coeffs /= max(la.norm(coeffs), 1e-15)
        H = build_hamiltonian(operators, coeffs)
        energy, variance, residual = energy_variance(psi, H)
        out.append(
            EHCSolution(
                eigenvalue=float(evals[k]),
                coefficients=coeffs,
                variance=variance,
                energy=energy,
                residual_norm=residual,
            )
        )
    return out


def ground_state_overlap(psi: np.ndarray, H: ArrayLikeOperator, *, k: int = 1) -> tuple[float, np.ndarray]:
    """
    Compare psi with the lowest-energy eigenspace of H.

    Returns the total overlap weight with the lowest k eigenvectors and the full
    dense spectrum.  This is useful because EHC guarantees eigenstate status,
    not ground-state status.
    """
    psi = normalize_state(psi)
    dense = H.toarray() if sp.issparse(H) else np.asarray(H)
    evals, evecs = la.eigh(dense)
    weight = float(np.sum(np.abs(evecs[:, :k].conj().T @ psi) ** 2))
    return weight, evals


def format_solution(
    solution: EHCSolution,
    operators: list[OperatorTerm],
    *,
    coefficient_cutoff: float = 1e-8,
) -> str:
    """Format a discovered Hamiltonian for terminal output."""
    lines = [
        f"QCM eigenvalue = {solution.eigenvalue:.6e}",
        f"variance       = {solution.variance:.6e}",
        f"residual norm  = {solution.residual_norm:.6e}",
        f"energy         = {solution.energy.real:.10f}",
        "coefficients:",
    ]
    for coeff, term in zip(solution.coefficients, operators, strict=True):
        if abs(coeff) >= coefficient_cutoff:
            lines.append(f"  {coeff:+.8f}  {term.name}")
    return "\n".join(lines)
