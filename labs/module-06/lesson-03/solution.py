"""Reference solution for lab 06.3 (combining changes)."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from frontierlab.stats import bootstrap_ci


def interaction(cells: dict) -> float:
    """2 x 2 interaction contrast of cell means: (AB − A) − (B − 0) = m11 − m10 − m01 + m00."""
    return cells[(1, 1)] - cells[(1, 0)] - cells[(0, 1)] + cells[(0, 0)]


def interaction_ci(per_window: dict, n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Interaction contrast per evaluation window, then a percentile bootstrap over windows (paired: every
    cell is scored on the same windows). per_window[(a, b)] is an array of per-window losses."""
    d = (np.asarray(per_window[(1, 1)]) - np.asarray(per_window[(1, 0)])
         - np.asarray(per_window[(0, 1)]) + np.asarray(per_window[(0, 0)]))
    return bootstrap_ci(d, n_boot=n_boot, seed=seed)


def additive_prediction(base: float, singles: dict) -> float:
    """What the combination would score if effects simply added: base + sum of (single − base)."""
    return base + sum(v - base for v in singles.values())


def design_cells(components: list[str], design: str) -> set[frozenset]:
    """Distinct cells (sets of components switched on) of a design.

    full: every subset (2^k cells). lite: the baseline, each component alone, the full combination and the
    combination with each component left out (at most 2k + 2 cells; fewer when they coincide, e.g. k = 2)."""
    comps = list(components)
    if design == "full":
        return {frozenset(c) for r in range(len(comps) + 1) for c in combinations(comps, r)}
    if design == "lite":
        cells = {frozenset(), frozenset(comps)}
        cells |= {frozenset([c]) for c in comps}
        cells |= {frozenset(set(comps) - {c}) for c in comps}
        return cells
    raise ValueError(design)
