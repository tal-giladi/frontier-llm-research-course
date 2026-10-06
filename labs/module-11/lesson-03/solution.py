"""Reference solution for lab 11.3."""

from __future__ import annotations

import math

import numpy as np


def interval(preds, level: float = 0.9) -> tuple[float, float]:
    lo, hi = np.quantile(np.asarray(preds, dtype=np.float64), [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def loss_at_fraction(curve, steps: int, f: float) -> float:
    s, v = min(curve, key=lambda sv: abs(sv[0] - f * steps))
    return math.nan if abs(s - f * steps) > 0.05 * steps + 1 else float(v)


def first_off_band(curve, steps: int, band, tol: float = 0.02, min_fraction: float = 0.1):
    fr = np.array([b["fraction"] for b in band], dtype=np.float64)
    hi = np.array([b["hi"] for b in band], dtype=np.float64)
    for s, v in sorted(curve):
        f = s / steps
        if f + 0.005 < min_fraction:
            continue
        if v > float(np.interp(f, fr, hi)) + tol:
            return int(s)
    return None


def lr_sensitivity(losses: dict, loss_init: float) -> float:
    v = np.array(list(losses.values()), dtype=np.float64)
    return float(np.mean(np.minimum(v, loss_init)) - v.min())


def go_no_go(checks: dict) -> str:
    failed = sorted(k for k, (ok, _) in checks.items() if not ok)
    return "go" if not failed else "no-go: " + ", ".join(failed)
