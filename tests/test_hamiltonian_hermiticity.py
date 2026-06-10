import numpy as np

from fqh_chern.hofstadter import single_particle_hofstadter


def test_single_particle_hofstadter_is_hermitian():
    H = single_particle_hofstadter(Lx=4, Ly=4, alpha=0.25, theta_x=0.3, theta_y=0.7)
    assert np.allclose(H, H.conj().T)
