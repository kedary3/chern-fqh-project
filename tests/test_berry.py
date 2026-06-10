import numpy as np

from fqh_chern.berry import abelian_chern, nonabelian_chern


def test_constant_state_has_zero_chern():
    states = np.zeros((4, 4, 2), dtype=complex)
    states[..., 0] = 1.0
    assert abs(abelian_chern(states)) < 1e-12


def test_constant_subspace_has_zero_chern():
    subspaces = np.zeros((4, 4, 1, 2), dtype=complex)
    subspaces[..., 0, 0] = 1.0
    assert abs(nonabelian_chern(subspaces)) < 1e-12
