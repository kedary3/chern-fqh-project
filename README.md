# Many-Body Chern Number Project for Fractional Quantum Hall Physics

This repository is a computational project for studying many-body Chern numbers in fractional quantum Hall-like lattice systems, especially interacting Harper-Hofstadter and fractional Chern-insulator models.

The project uses:

- **QuSpin** for exact diagonalization of many-body lattice Hamiltonians.
- **QuTiP** for small-system validation, state/operator utilities, and open-system toy models.
- **OQuPy** for later-stage non-Markovian open-system extensions.

The core calculation is the many-body Chern number obtained from twisted boundary conditions:

```text
(theta_x, theta_y) in [0, 2 pi) x [0, 2 pi)
```

At every twist point, the many-body Hamiltonian is diagonalized. The ground state or ground-state multiplet is used to compute Berry curvature on the twist-angle torus using the Fukui-Hatsugai-Suzuki lattice formula.

## Physics target

The baseline Hamiltonian is an interacting Harper-Hofstadter model on an `Lx x Ly` torus:

```text
H = -t sum_{x,y} [
      exp(i theta_x/Lx) c^dagger_{x+1,y} c_{x,y}
    + exp(i(2 pi alpha x + theta_y/Ly)) c^dagger_{x,y+1} c_{x,y}
    + h.c.
    ]
    + V sum_{<i,j>} n_i n_j.
```

The project searches for fractional Chern-insulator behavior through:

1. A nearly degenerate low-energy ground-state manifold.
2. A finite many-body gap above that manifold.
3. A stable non-Abelian many-body Chern number.
4. Smooth convergence under twist-grid refinement.
5. Robustness under weak disorder or interaction variation.

## Repository layout

```text
chern-fqh-project/
├── README.md
├── pyproject.toml
├── requirements.txt
├── src/fqh_chern/
│   ├── __init__.py
│   ├── lattice.py
│   ├── hofstadter.py
│   ├── interactions.py
│   ├── diagonalization.py
│   ├── berry.py
│   ├── observables.py
│   ├── qutip_validation.py
│   ├── oqupy_dynamics.py
│   └── plotting.py
├── scripts/
│   ├── run_spectrum.py
│   ├── run_chern_scan.py
│   └── run_oqupy_noise_scan.py
├── notebooks/
│   ├── 01_single_particle_hofstadter.ipynb
│   ├── 02_many_body_spectrum.ipynb
│   ├── 03_many_body_chern_number.ipynb
│   └── 04_open_system_chern_response.ipynb
└── tests/
    ├── test_berry.py
    ├── test_hamiltonian_hermiticity.py
    └── test_chern_integer.py
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

For QuSpin, installation can be platform-sensitive. If `pip install quspin` fails, consult the QuSpin installation notes for your platform.

## Minimal usage

Run a single-particle Hofstadter benchmark:

```bash
python scripts/run_spectrum.py --Lx 4 --Ly 4 --alpha 0.25
```

Run a many-body Chern scan:

```bash
python scripts/run_chern_scan.py --Lx 4 --Ly 4 --N 2 --alpha 0.25 --V 2.0 --grid 6 --multiplet 1
```

Run tests:

```bash
pytest
```

## Notes on library roles

The closed-system many-body Chern calculation should be driven primarily by QuSpin and NumPy/SciPy. QuTiP is included for validation and small-system open-system checks. OQuPy is intentionally placed in an extension module because process-tensor simulations are most useful after the closed-system topological diagnostic has been validated.

## Suggested development roadmap

1. Validate the single-particle Hofstadter Hamiltonian.
2. Validate many-body basis construction in fixed particle-number sectors.
3. Compute low-energy spectra over a twist-angle grid.
4. Compute Abelian and non-Abelian Chern numbers.
5. Scan interaction strength and flux density.
6. Add disorder and finite-size scaling.
7. Add QuTiP and OQuPy open-system extensions.
