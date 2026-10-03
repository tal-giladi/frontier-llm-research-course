"""Reference solution for lab 01.4."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as st


def mde(seed_std, n_per_arm, alpha=0.05, power=0.8):
    z = st.norm.ppf(1 - alpha / 2) + st.norm.ppf(power)
    return z * seed_std * math.sqrt(2.0 / n_per_arm)


def seeds_needed(seed_std, effect, alpha=0.05, power=0.8, max_n=1000):
    for n in range(2, max_n + 1):
        if mde(seed_std, n, alpha, power) <= effect:
            return n
    raise ValueError(f"more than {max_n} seeds per arm needed")


def unpaired_bootstrap(a, b, n_boot=10000, alpha=0.05, seed=0):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, a.size, size=(n_boot, a.size))
    ib = rng.integers(0, b.size, size=(n_boot, b.size))
    boots = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": float(a.mean() - b.mean()), "ci": (float(lo), float(hi))}


def seed_level_ci(base, new, paired, alpha=0.05):
    b, n = np.asarray(base, dtype=np.float64), np.asarray(new, dtype=np.float64)
    if paired:
        if b.shape != n.shape:
            raise ValueError("paired needs one new run per base seed")
        d = n - b
        se = d.std(ddof=1) / math.sqrt(d.size)
        df = d.size - 1
        m = d.mean()
    else:
        vb, vn = b.var(ddof=1) / b.size, n.var(ddof=1) / n.size
        se = math.sqrt(vb + vn)
        df = (vb + vn) ** 2 / (vb**2 / (b.size - 1) + vn**2 / (n.size - 1))
        m = n.mean() - b.mean()
    t = st.t.ppf(1 - alpha / 2, df)
    return {"mean_diff": float(m), "ci": (float(m - t * se), float(m + t * se)), "se": float(se)}
