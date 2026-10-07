"""Lab 14.2 — the decision machinery of a controlled objective comparison. Fill in the TODOs; run
`pytest labs/module-14/lesson-02` to check. ``compare_lab.py`` uses these functions to tune, compare and decide.
"""

from __future__ import annotations

import math

# Stated before any run (the experiment contract). compare_lab.py prints them with the results.
THRESHOLDS = {"collapse_drawdown": 0.10, "grad_spikes": 3, "entropy_drop": 0.25, "ratio_max": 50.0}


def select_lr(tuning: dict[float, dict]) -> float:
    """Pick one learning rate per objective from its tuning runs.

    ``tuning`` maps lr -> {"score": mean training pass rate over the last 25 steps (tuning seed, training
    prompts only), "collapsed": bool}. Rule: the highest score among runs that did not collapse; on a tie
    (scores within 0.005 of each other) the smaller lr. If every run collapsed, the smallest lr.
    """
    raise NotImplementedError("TODO 1: learning-rate selection")


def paired_interval(a: list[float], b: list[float], alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, low, high) of the per-seed differences a[s] - b[s] with a two-sided t-interval
    (n - 1 degrees of freedom, scipy.stats.t.ppf). With one seed return (d, nan, nan)."""
    raise NotImplementedError("TODO 2: paired t-interval")


def verdict(vs_baseline: tuple[float, float, float], vs_control: tuple[float, float, float]) -> str:
    """Apply the decision rule to two paired intervals (mean, low, high).

    Returns one of: "below control" (the arm is not shown to beat the random-reward control: low <= 0),
    else "better than baseline" (low > 0), "worse than baseline" (high < 0) or "inconclusive".
    The control check comes first: an arm that does not beat the control says nothing about objectives.
    """
    raise NotImplementedError("TODO 3: the decision rule")


def stability_flags(st: dict, thresholds: dict = THRESHOLDS) -> list[str]:
    """Names of the pre-stated stability thresholds a run breaks, in this order:
    "collapse" (eval_drawdown >= collapse_drawdown), "grad_spikes" (grad_spikes >= grad_spikes),
    "entropy_collapse" (entropy_drop >= entropy_drop), "ratio_blowup" (ratio_max >= ratio_max).
    A NaN value never breaks a threshold."""
    raise NotImplementedError("TODO 4: stability flags")
