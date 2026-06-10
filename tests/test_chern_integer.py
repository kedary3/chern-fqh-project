import numpy as np

from fqh_chern.berry import abelian_chern


def test_trivial_chern_is_integer():
    states = np.zeros((5, 5, 3), dtype=complex)
    states[..., 1] = 1.0
    c = abelian_chern(states)
    assert np.isclose(c, round(c))
