"""Lab 11.3 — de-risking a run. Fill in the TODOs; run `pytest labs/module-11/lesson-03` to check.

References: ``frontierlab.scaling.derisk`` and ``frontierlab.scaling.fit``.
"""

from __future__ import annotations

import numpy as np


def interval(preds, level: float = 0.9) -> tuple[float, float]:
    """Central ``level`` interval of bootstrap predictions: the (1−level)/2 and (1+level)/2 quantiles (np.quantile)."""
    raise NotImplementedError("TODO 1: a bootstrap prediction interval")


def loss_at_fraction(curve, steps: int, f: float) -> float:
    """``curve``: list of (step, val_loss). Return the loss of the evaluation whose step is nearest to f·steps,
    or nan if that evaluation is more than 0.05·steps + 1 steps away."""
    raise NotImplementedError("TODO 2: read a run at a fraction of its schedule")


def first_off_band(curve, steps: int, band, tol: float = 0.02, min_fraction: float = 0.1):
    """``band``: list of dicts with ``fraction`` and ``hi`` (sorted by fraction). Walk the evaluations in step order;
    skip those with step/steps + 0.005 < ``min_fraction`` (so the evaluation at step steps//10 counts as the
    10% point); return the first step whose loss exceeds the band's upper edge at
    that fraction (np.interp between band fractions) by more than ``tol``. Return None if there is none."""
    raise NotImplementedError("TODO 3: the off-band stopping rule")


def lr_sensitivity(losses: dict, loss_init: float) -> float:
    """Wortsman et al. (arXiv 2309.14322, section 2.2): mean over the swept learning rates of
    min(loss(lr), loss_init) − min over the sweep of loss(lr). ``losses`` maps lr -> final loss."""
    raise NotImplementedError("TODO 4: learning-rate sensitivity")


def go_no_go(checks: dict) -> str:
    """``checks``: name -> (passed: bool, detail: str). "go" if every check passed, else "no-go: " followed by the
    failed names, sorted, joined by ", "."""
    raise NotImplementedError("TODO 5: the launch decision")
