"""Paired uncertainty for a capstone comparison, and the decision rule (lesson 20.1).

A capstone comparison has two sources of noise: **seeds** (initialisation and data order) and **evaluation items**
(held-out windows, prompts). Compared arms share both: seed ``s`` of arm A and seed ``s`` of arm B start from the
same initial weights and see the same data order, and both are scored on the same items. So the natural unit is the
paired difference ``d[s, w] = a[s, w] - b[s, w]`` and the interval should resample both levels:

* :func:`hierarchical_bootstrap` — resample seeds with replacement, then items within each chosen seed, and take the
  mean of ``d`` each time (two-level, or hierarchical, bootstrap; METR uses the same idea for its time horizons);
* :func:`seed_t_interval` — the paired t-interval over the per-seed means (only the seed level; with 3 seeds the
  t quantile is 4.30, so it is wide). Report both; with 2-3 seeds the bootstrap's seed level is coarse (only
  ``C(2S-1, S)`` distinct seed resamples, 10 for S = 3) and tends to be narrower than the truth.
* :func:`decide` — the pre-stated rule: *equivalent* if the whole interval lies inside ``[-margin, +margin]``,
  otherwise *a_lower* / *a_higher* if it excludes zero, otherwise *inconclusive*.
* :func:`noise_floor` — the baseline's seed standard deviation and the minimum detectable effect for the design.
"""

from __future__ import annotations

import math

import numpy as np

from frontierlab.stats import min_detectable_effect

DECISIONS = ("equivalent", "a_lower", "a_higher", "inconclusive")


def hierarchical_bootstrap(a, b, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Paired two-level bootstrap of mean(a - b) for arrays of shape (seeds, items).

    Each resample draws ``S`` seed indices with replacement, then for each drawn seed ``W`` item indices with
    replacement, and averages the paired differences. Returns ``{"mean_diff", "ci": (lo, hi), "n_seeds",
    "n_items", "method"}``.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("need two (seeds, items) arrays with the same seeds and items in the same order")
    d = a - b
    S, W = d.shape
    rng = np.random.default_rng(seed)
    s_idx = rng.integers(0, S, size=(n_boot, S))
    w_idx = rng.integers(0, W, size=(n_boot, S, W))
    boots = d[s_idx[:, :, None], w_idx].mean(axis=(1, 2))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": float(d.mean()), "ci": (float(lo), float(hi)), "n_seeds": S, "n_items": W,
            "method": f"paired hierarchical bootstrap over {S} seeds and {W} items ({n_boot} resamples)"}


def seed_t_interval(a_by_seed, b_by_seed, alpha: float = 0.05) -> dict:
    """Paired t-interval of the per-seed differences (seed level only)."""
    from scipy.stats import t
    d = np.asarray(a_by_seed, dtype=np.float64) - np.asarray(b_by_seed, dtype=np.float64)
    m = float(d.mean())
    if d.size < 2:
        return {"mean_diff": m, "ci": (float("nan"), float("nan")), "n_seeds": int(d.size),
                "method": "paired t over seed means (needs at least 2 seeds)"}
    h = float(t.ppf(1 - alpha / 2, d.size - 1) * d.std(ddof=1) / math.sqrt(d.size))
    return {"mean_diff": m, "ci": (m - h, m + h), "n_seeds": int(d.size),
            "method": f"paired t over {d.size} seed means"}


def decide(mean: float, lo: float, hi: float, margin: float) -> str:
    """The capstone decision rule, stated before the runs. ``mean`` is a - b.

    * ``equivalent`` — the interval lies inside [-margin, +margin] (a difference too small to matter, even if it
      excludes zero);
    * ``a_lower`` / ``a_higher`` — otherwise, the interval excludes zero on that side;
    * ``inconclusive`` — otherwise (including a NaN interval).
    """
    if any(math.isnan(x) for x in (mean, lo, hi)):
        return "inconclusive"
    if lo >= -margin and hi <= margin:
        return "equivalent"
    if hi < 0:
        return "a_lower"
    if lo > 0:
        return "a_higher"
    return "inconclusive"


def noise_floor(baseline_by_seed, n_per_arm: int | None = None) -> dict:
    """Seed std of the baseline's per-seed metric and the minimum detectable effect (80% power, alpha 0.05)."""
    x = np.asarray(baseline_by_seed, dtype=np.float64)
    sd = float(x.std(ddof=1)) if x.size > 1 else float("nan")
    n = n_per_arm or int(x.size)
    return {"seed_std": sd, "n_seeds": int(x.size), "mde": min_detectable_effect(sd, n) if x.size > 1 else float("nan")}


__all__ = ["DECISIONS", "decide", "hierarchical_bootstrap", "noise_floor", "seed_t_interval"]
