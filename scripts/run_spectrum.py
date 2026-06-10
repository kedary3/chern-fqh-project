#!/usr/bin/env python
"""Run a single-particle Hofstadter spectrum benchmark."""

from __future__ import annotations

import argparse
import numpy as np

from fqh_chern.hofstadter import single_particle_hofstadter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--Lx", type=int, default=4)
    parser.add_argument("--Ly", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--theta-x", type=float, default=0.0)
    parser.add_argument("--theta-y", type=float, default=0.0)
    args = parser.parse_args()

    H = single_particle_hofstadter(args.Lx, args.Ly, args.alpha, args.theta_x, args.theta_y)
    evals = np.linalg.eigvalsh(H)
    print("Eigenvalues:")
    print(np.array2string(evals, precision=8, suppress_small=True))


if __name__ == "__main__":
    main()
