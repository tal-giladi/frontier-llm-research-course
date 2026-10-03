"""Lab 01.4 — uncertainty. Fill in the TODOs; run `pytest labs/module-01/lesson-04`.

Plain NumPy/SciPy so you can see every step. You may use ``frontierlab.stats.paired_bootstrap`` and
``frontierlab.stats.holm`` (lesson 01.4 explains them) but not ``min_detectable_effect``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as st


def mde(seed_std: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable effect between two independent arms with n seeds each (normal approximation).

    MDE = (z_{1-alpha/2} + z_{power}) * seed_std * sqrt(2 / n_per_arm). Use scipy.stats.norm.ppf for z.
    """
    raise NotImplementedError("TODO 1: minimum detectable effect")


def seeds_needed(seed_std: float, effect: float, alpha: float = 0.05, power: float = 0.8, max_n: int = 1000) -> int:
    """Smallest n per arm whose MDE is <= effect (use your ``mde``)."""
    raise NotImplementedError("TODO 2: seeds needed for a planned ablation")


def unpaired_bootstrap(a, b, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> dict:
    """CI of mean(a) - mean(b) when a and b are resampled INDEPENDENTLY (ignores that items are shared).

    Draw ``n_boot`` resamples of a and of b (with replacement, separate index draws from one
    ``np.random.default_rng(seed)``), take the difference of means each time, and return
    {"mean_diff": ..., "ci": (lo, hi)} with the alpha/2 and 1-alpha/2 quantiles.
    """
    raise NotImplementedError("TODO 3: unpaired bootstrap")


def seed_level_ci(base, new, paired: bool, alpha: float = 0.05) -> dict:
    """t-interval for mean(new) - mean(base) over SEEDS (one number per run).

    paired=True: base[i] and new[i] share seed i (same init and data order). Use d = new - base,
    CI = mean(d) +- t_{n-1} * sd(d) / sqrt(n)   (sd with ddof=1).
    paired=False: Welch interval, se = sqrt(var_b/n_b + var_n/n_n), degrees of freedom by
    Welch–Satterthwaite. Return {"mean_diff", "ci": (lo, hi), "se"}.
    """
    raise NotImplementedError("TODO 4: seed-level interval, paired and unpaired")
