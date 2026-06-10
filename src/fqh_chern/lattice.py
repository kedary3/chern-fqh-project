"""Lattice indexing utilities."""

from __future__ import annotations


def site_index(x: int, y: int, Lx: int, Ly: int) -> int:
    """Map a two-dimensional lattice coordinate to a one-dimensional site index."""
    if not (0 <= x < Lx and 0 <= y < Ly):
        raise ValueError("Coordinates must satisfy 0 <= x < Lx and 0 <= y < Ly.")
    return x + Lx * y


def coords_from_site(site: int, Lx: int, Ly: int) -> tuple[int, int]:
    """Map a one-dimensional site index back to two-dimensional coordinates."""
    if not (0 <= site < Lx * Ly):
        raise ValueError("Site index out of range.")
    y, x = divmod(site, Lx)
    return x, y


def nearest_neighbor_pairs(Lx: int, Ly: int) -> list[tuple[int, int]]:
    """Return unique nearest-neighbor pairs on a periodic rectangular lattice."""
    pairs: set[tuple[int, int]] = set()
    for x in range(Lx):
        for y in range(Ly):
            i = site_index(x, y, Lx, Ly)
            for xp, yp in [((x + 1) % Lx, y), (x, (y + 1) % Ly)]:
                j = site_index(xp, yp, Lx, Ly)
                pairs.add(tuple(sorted((i, j))))
    return sorted(pairs)
