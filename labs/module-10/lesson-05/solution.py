"""Reference solution for lab 10.5 — continued training and forgetting."""

from __future__ import annotations

import numpy as np


def gain_and_forgetting(target_arm, target_ctrl, general_arm, general_ctrl):
    """Per-seed (gain, forgetting) against the equal-token control, from mean held-out losses.
    gain = ctrl − arm on the target domain (positive = better); forgetting = arm − ctrl elsewhere (positive = worse)."""
    gain = np.asarray(target_ctrl, dtype=np.float64) - np.asarray(target_arm, dtype=np.float64)
    forg = np.asarray(general_arm, dtype=np.float64) - np.asarray(general_ctrl, dtype=np.float64)
    return gain, forg


def target_tokens(steps, batch, seq, frac):
    return steps * batch * seq * frac


def pareto(points):
    """Indices of the arms not dominated in (higher gain, lower forgetting); points = [(gain, forgetting), ...]."""
    keep = []
    for i, (g, f) in enumerate(points):
        if not any((g2 >= g and f2 <= f) and (g2 > g or f2 < f) for j, (g2, f2) in enumerate(points) if j != i):
            keep.append(i)
    return keep


def choose(arms_ci, max_forgetting):
    """arms_ci: {name: {"gain": (lo, hi), "forgetting": (lo, hi), "gain_mean": m}}. Among arms whose gain lower
    bound > 0 and forgetting upper bound <= max_forgetting, the one with the largest mean gain; None if none."""
    ok = {k: v for k, v in arms_ci.items() if v["gain"][0] > 0 and v["forgetting"][1] <= max_forgetting}
    return max(ok, key=lambda k: ok[k]["gain_mean"]) if ok else None
