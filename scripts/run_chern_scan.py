#!/usr/bin/env python
"""Run a baseline single-particle Chern scan over twist angles.

The many-body QuSpin implementation is the next step. This script validates the
Berry-curvature machinery on dense single-particle Hofstadter data.
"""

from __future__ import annotations

import argparse

from fqh_chern.berry import abelian_chern
from fqh_chern.diagonalization import single_particle_twist_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--Lx", type=int, default=4)
    parser.add_argument("--Ly", type=int, default=4)
    parser.add_argument("--N", type=int, default=1, help="Reserved for future many-body QuSpin implementation.")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--V", type=float, default=0.0, help="Reserved for future interaction scan.")
    parser.add_argument("--grid", type=int, default=6)
    parser.add_argument("--multiplet", type=int, default=1)
    parser.add_argument("--band-index", type=int, default=0)
    args = parser.parse_args()

    states = single_particle_twist_grid(
        args.Lx, args.Ly, args.alpha, args.grid, band_index=args.band_index
    )
    chern = abelian_chern(states)
    print(f"Abelian Chern estimate: {chern:.12f}")


if __name__ == "__main__":
    main()
