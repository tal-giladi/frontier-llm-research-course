"""Reference solution for lab 01.3."""

from __future__ import annotations

from frontierlab.flops import flops_per_token
from frontierlab.model.config import ModelConfig


def matched_steps(axis, ref: ModelConfig, other: ModelConfig, ref_steps: int, T: int,
                  ref_tok_per_s=None, other_tok_per_s=None) -> int:
    if axis == "tokens":
        return ref_steps
    if axis == "flops":
        return round(ref_steps * flops_per_token(ref, T) / flops_per_token(other, T))
    if axis == "wallclock":
        if not ref_tok_per_s or not other_tok_per_s:
            raise ValueError("equal wall-clock needs both measured throughputs on the same hardware")
        return round(ref_steps * other_tok_per_s / ref_tok_per_s)
    if axis == "params":
        raise ValueError("equal parameters fixes the models; pick tokens, flops or wallclock for the budget")
    raise ValueError(f"unknown axis {axis!r}")


def select_then_report(val_scores: dict, test_scores: dict):
    best = min(val_scores, key=val_scores.get)
    return best, test_scores[best]


def decide(ci, min_gain: float) -> str:
    lo, hi = ci
    if hi < -min_gain:
        return "adopt"
    if lo > -min_gain:
        return "reject"
    return "inconclusive"
