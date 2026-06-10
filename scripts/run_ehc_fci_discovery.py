#!/usr/bin/env python
"""Run EHC inverse Hamiltonian discovery for a small FQHE/FCI lattice target.

The target state is the exact ground state of an interacting Harper-Hofstadter
model.  The program then forgets the couplings and reconstructs low-variance
candidate Hamiltonians from a chosen operator library using the quantum
covariance matrix inverse method of Chertkov and Clark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as la

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataclasses import replace

from fqh_chern.inverse_ehc import build_hamiltonian, discover_hamiltonians, format_solution, ground_state_overlap
from fqh_chern.many_body import (
    expanded_fci_operator_library,
    grouped_fci_operator_library,
    hofstadter_many_body_hamiltonian_dense,
    quspin_grouped_fci_operator_library,
    quspin_hofstadter_many_body_hamiltonian_dense,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Lx", type=int, default=4)
    parser.add_argument("--Ly", type=int, default=3)
    parser.add_argument("--N", type=int, default=2, help="number of spinless fermions")
    parser.add_argument("--alpha", type=float, default=1 / 3, help="flux per plaquette")
    parser.add_argument("--theta-x", type=float, default=0.0)
    parser.add_argument("--theta-y", type=float, default=0.0)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--V", type=float, default=2.0)
    parser.add_argument(
        "--library",
        choices=["grouped", "expanded"],
        default="grouped",
        help="operator library used by EHC",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "dense", "quspin"],
        default="auto",
        help="Hamiltonian-construction backend; auto prefers QuSpin when installed",
    )
    parser.add_argument("--solutions", type=int, default=5)
    parser.add_argument("--cutoff", type=float, default=1e-7)
    parser.add_argument("--save", type=Path, default=None, help="optional .npz output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    backend = args.backend
    if backend == "auto":
        try:
            import quspin  # noqa: F401

            backend = "quspin"
        except Exception:
            backend = "dense"

    if backend == "quspin":
        try:
            H_target, basis = quspin_hofstadter_many_body_hamiltonian_dense(
                args.Lx,
                args.Ly,
                args.N,
                args.alpha,
                theta_x=args.theta_x,
                theta_y=args.theta_y,
                t=args.t,
                V=args.V,
            )
        except Exception as exc:
            if args.backend == "quspin":
                raise
            print(f"QuSpin backend unavailable or failed ({exc}); falling back to dense backend.")
            backend = "dense"

    if backend == "dense":
        H_target, basis = hofstadter_many_body_hamiltonian_dense(
            args.Lx,
            args.Ly,
            args.N,
            args.alpha,
            theta_x=args.theta_x,
            theta_y=args.theta_y,
            t=args.t,
            V=args.V,
        )
    evals, evecs = la.eigh(H_target)
    psi_target = evecs[:, 0]

    if backend == "quspin" and args.library == "grouped":
        operators, _ = quspin_grouped_fci_operator_library(
            args.Lx,
            args.Ly,
            args.N,
            args.alpha,
            theta_x=args.theta_x,
            theta_y=args.theta_y,
            include_onsite=True,
        )
    elif args.library == "grouped":
        operators, _ = grouped_fci_operator_library(
            args.Lx,
            args.Ly,
            args.N,
            args.alpha,
            theta_x=args.theta_x,
            theta_y=args.theta_y,
            include_onsite=True,
        )
    else:
        if backend == "quspin":
            print("Expanded library currently uses the dense reference backend for individual local operators.")
        operators, _ = expanded_fci_operator_library(
            args.Lx,
            args.Ly,
            args.N,
            args.alpha,
            theta_x=args.theta_x,
            theta_y=args.theta_y,
        )

    solutions = discover_hamiltonians(psi_target, operators, n_solutions=args.solutions)

    print("EHC inverse Hamiltonian discovery for a lattice FQHE/FCI target")
    print("=" * 72)
    print(f"system                 : {args.Lx} x {args.Ly}, N={args.N}, dim={len(basis.states)}")
    print(f"flux alpha             : {args.alpha}")
    print(f"backend                : {backend}")
    print(f"target couplings       : t={args.t}, V={args.V}")
    print(f"target ground energy   : {evals[0]:.10f}")
    if len(evals) > 1:
        print(f"target many-body gap   : {evals[1] - evals[0]:.10e}")
    print(f"operator library       : {args.library} ({len(operators)} operators)")
    print()

    for i, sol in enumerate(solutions):
        H_candidate = build_hamiltonian(operators, sol.coefficients)
        gs_weight, spectrum = ground_state_overlap(psi_target, H_candidate, k=1)
        flipped_weight, flipped_spectrum = ground_state_overlap(psi_target, -H_candidate, k=1)

        display_sol = sol
        display_spectrum = spectrum
        display_weight = gs_weight
        orientation_note = "as returned by QCM"
        if flipped_weight > gs_weight:
            display_sol = replace(sol, coefficients=-sol.coefficients, energy=-sol.energy)
            display_spectrum = flipped_spectrum
            display_weight = flipped_weight
            orientation_note = "sign flipped to maximize target ground-state overlap"

        print(f"Candidate {i}")
        print("-" * 72)
        print(format_solution(display_sol, operators, coefficient_cutoff=args.cutoff))
        print(f"orientation                 = {orientation_note}")
        print(f"ground-state overlap weight = {display_weight:.10f}")
        if len(display_spectrum) > 1:
            print(f"candidate gap              = {display_spectrum[1] - display_spectrum[0]:.10e}")
        print()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save,
            target_evals=evals,
            target_state=psi_target,
            solution_eigenvalues=np.array([s.eigenvalue for s in solutions]),
            solution_coefficients=np.vstack([s.coefficients for s in solutions]),
            operator_names=np.array([op.name for op in operators]),
        )
        print(f"Saved EHC results to {args.save}")


if __name__ == "__main__":
    main()
