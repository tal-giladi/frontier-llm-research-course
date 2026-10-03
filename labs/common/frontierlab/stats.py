"""Uncertainty for ML experiments (Module 1, lesson 01.4).

Everything here is plain NumPy so learners can read it. The functions answer three questions:

* How noisy is a number?  -> :func:`summary`, :func:`bootstrap_ci`
* Is A better than B on the same evaluation items?  -> :func:`paired_bootstrap`
* Could my planned experiment detect the effect I care about?  -> :func:`min_detectable_effect`
"""

from __future__ import annotations

import math

import numpy as np


def summary(values) -> dict:
    """Mean, sample std (ddof=1), standard error and n of a list of per-seed results."""
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    return {"n": n, "mean": float(x.mean()), "std": sd, "sem": sd / math.sqrt(n) if n > 1 else float("nan")}


def bootstrap_ci(values, stat=np.mean, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap confidence interval of ``stat`` over ``values`` (items or seeds)."""
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = np.apply_along_axis(stat, 1, x[idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(stat(x)), float(lo), float(hi)


def paired_bootstrap(a, b, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Mean of ``a - b`` over paired items (same eval windows / prompts) with a bootstrap CI.

    Pairing removes the item-difficulty variance both systems share, which is why it detects
    smaller differences than comparing two unpaired means. ``p_le_zero`` is the fraction of
    bootstrap means <= 0 (a one-sided bootstrap p-value for "a > b").
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired comparison needs the same items in the same order")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": float(d.mean()), "ci": (float(lo), float(hi)), "p_le_zero": float((boots <= 0).mean())}


def min_detectable_effect(seed_std: float, n_per_arm: int, z_alpha: float = 1.96, z_power: float = 0.84) -> float:
    """Smallest true difference between two arms detectable with ~80% power at alpha 0.05 (two-sided).

    Normal approximation for two independent arms with equal seed std:
    ``MDE = (z_alpha + z_power) * seed_std * sqrt(2 / n_per_arm)``.
    """
    return (z_alpha + z_power) * seed_std * math.sqrt(2.0 / n_per_arm)


def holm(pvalues) -> list[float]:
    """Holm-Bonferroni adjusted p-values (for several comparisons against one baseline)."""
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    m, adj, running = p.size, np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
