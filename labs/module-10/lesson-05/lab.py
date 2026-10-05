"""Lab 10.5 — continued training and forgetting. Fill in the TODOs; run `pytest labs/module-10/lesson-05` to check."""

from __future__ import annotations

import numpy as np


def gain_and_forgetting(target_arm, target_ctrl, general_arm, general_ctrl) -> tuple[np.ndarray, np.ndarray]:
    """Per-seed arrays (gain, forgetting) against the equal-token control, from mean held-out losses per seed.

    gain = ctrl − arm on the target domain (positive = the arm is better there);
    forgetting = arm − ctrl on the general held-out set (positive = the arm is worse there).
    """
    raise NotImplementedError("TODO 1: gain and forgetting, both against the control")


def target_tokens(steps: int, batch: int, seq: int, frac: float) -> float:
    """Tokens of the target domain a continued-training run sees (exact with the counter-based mixture sampler
    when frac is a multiple of 1/block)."""
    raise NotImplementedError("TODO 2: target-token budget")


def pareto(points) -> list[int]:
    """Indices of arms not dominated in (higher gain, lower forgetting). Arm j dominates arm i if
    gain_j >= gain_i and forgetting_j <= forgetting_i with at least one strict."""
    raise NotImplementedError("TODO 3: the trade-off frontier")


def choose(arms_ci: dict, max_forgetting: float):
    """The pre-stated rule. ``arms_ci[name] = {"gain": (lo, hi), "forgetting": (lo, hi), "gain_mean": m}``.
    Eligible: gain lower bound > 0 and forgetting upper bound <= max_forgetting. Return the eligible arm with
    the largest mean gain, or None if no arm is eligible."""
    raise NotImplementedError("TODO 4: choose the replay fraction")
