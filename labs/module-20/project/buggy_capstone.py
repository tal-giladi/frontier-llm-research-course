"""A colleague's version of the capstone analysis pieces (Module 20 project, debugging task).

Their message: "Rewrote the analysis to be simpler and faster. The QK-Clip reproduction now comes out 'a higher'
instead of 'equivalent', so clipping does cost loss after all, and the QK-norm extension's interval is much tighter
now. Can you sign off on the report today?"

Three bugs. Each changes what the report concludes. Find them by the sentence of the report they change, with the
test that isolates each, before you look at the fix.
"""

from __future__ import annotations

import math

import numpy as np


def pair(losses_a: dict[int, list[float]], losses_b: dict[int, list[float]]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    a = np.array(list(losses_a.values()), dtype=np.float64)
    b = np.array(list(losses_b.values()), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("the two arms were evaluated on different numbers of items")
    return a, b, sorted(losses_a)


def interval(a, b, n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    flat = d.reshape(-1)
    idx = rng.integers(0, flat.size, size=(n_boot, flat.size))
    boots = flat[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(d.mean()), float(lo), float(hi)


def decide(mean: float, lo: float, hi: float, margin: float) -> str:
    if any(math.isnan(x) for x in (mean, lo, hi)):
        return "inconclusive"
    if hi < 0:
        return "a_lower"
    if lo > 0:
        return "a_higher"
    if lo >= -margin and hi <= margin:
        return "equivalent"
    return "inconclusive"
