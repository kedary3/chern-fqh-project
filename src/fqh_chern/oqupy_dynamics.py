"""OQuPy extension hooks for non-Markovian open-system studies."""

from __future__ import annotations


def describe_oqupy_extension() -> str:
    """Return the intended OQuPy role in the project."""
    return (
        "Use OQuPy after the closed-system Chern-number calculation is stable. "
        "A natural extension is to couple an effective low-energy Hall manifold "
        "to a structured bath and monitor decoherence of Berry-phase or pumped-charge diagnostics."
    )


def build_process_tensor_model(*args, **kwargs):
    """Placeholder for an OQuPy process-tensor model."""
    raise NotImplementedError(
        "Define the reduced low-energy system, bath spectral density, and system-bath coupling first."
    )
