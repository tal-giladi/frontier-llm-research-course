"""The three analysis pieces of a capstone (Module 20 project): pairing, interval, decision.

``capstone_project.py`` and ``test_pieces.py`` load this file, or ``buggy_capstone.py`` when ``CAPSTONE=buggy``.
"""

from __future__ import annotations

import math

import numpy as np


def pair(losses_a: dict[int, list[float]], losses_b: dict[int, list[float]]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Two (seeds, items) arrays paired by seed: row i of both comes from the same seed. Only shared seeds."""
    seeds = sorted(set(losses_a) & set(losses_b))
    a = np.array([losses_a[s] for s in seeds], dtype=np.float64)
    b = np.array([losses_b[s] for s in seeds], dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("the two arms were evaluated on different numbers of items")
    return a, b, seeds


def interval(a, b, n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of a - b and its 95% paired hierarchical bootstrap interval (seeds, then items)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    S, W = d.shape
    rng = np.random.default_rng(seed)
    s_idx = rng.integers(0, S, size=(n_boot, S))
    w_idx = rng.integers(0, W, size=(n_boot, S, W))
    boots = d[s_idx[:, :, None], w_idx].mean(axis=(1, 2))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(d.mean()), float(lo), float(hi)


def decide(mean: float, lo: float, hi: float, margin: float) -> str:
    """equivalent (interval inside the margin) first, then a_lower / a_higher, else inconclusive."""
    if any(math.isnan(x) for x in (mean, lo, hi)):
        return "inconclusive"
    if lo >= -margin and hi <= margin:
        return "equivalent"
    if hi < 0:
        return "a_lower"
    if lo > 0:
        return "a_higher"
    return "inconclusive"
