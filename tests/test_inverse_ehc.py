import numpy as np
import scipy.linalg as la

from fqh_chern.inverse_ehc import OperatorTerm, build_hamiltonian, discover_hamiltonians, ground_state_overlap


def test_ehc_recovers_two_level_parent_hamiltonian():
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    H = 0.3 * sx - 1.2 * sz
    evals, evecs = la.eigh(H)
    psi = evecs[:, 0]

    ops = [OperatorTerm("sx", sx), OperatorTerm("sz", sz)]
    solutions = discover_hamiltonians(psi, ops, n_solutions=1)

    assert solutions[0].eigenvalue < 1e-12
    assert solutions[0].variance < 1e-12

    H_found = build_hamiltonian(ops, solutions[0].coefficients)
    overlap, _ = ground_state_overlap(psi, H_found)
    assert overlap > 1.0 - 1e-12
