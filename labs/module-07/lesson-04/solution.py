"""Reference solution for lab 07.4."""

from __future__ import annotations

import math


def wsd_lr(step, lr, warmup, decay_start, decay_steps, shape="linear", min_ratio=0.0):
    if step < warmup:
        return lr * (step + 1) / warmup
    if step < decay_start:
        return lr
    u = min(1.0, (step - decay_start + 1) / max(1, decay_steps))
    if u >= 1.0:
        return lr * min_ratio
    f = 1 - u if shape == "linear" else 1 - math.sqrt(u)
    return lr * (min_ratio + (1 - min_ratio) * f)


def branch_plan(totals, decay_frac):
    out = []
    for S in totals:
        d = max(1, round(S * decay_frac))
        out.append({"total": S, "decay_start": S - d, "decay_steps": d})
    return out


def branch_cost(plan):
    wsd = max(p["decay_start"] for p in plan) + sum(p["decay_steps"] for p in plan)
    cos = sum(p["total"] for p in plan)
    return {"wsd_steps": wsd, "cosine_steps": cos, "saving": 1 - wsd / cos}


def decay_drop(steps, losses, decay_start, window=10):
    before = [l for s, l in zip(steps, losses) if s <= decay_start][-window:]
    return sum(before) / len(before) - sum(losses[-window:]) / len(losses[-window:])
