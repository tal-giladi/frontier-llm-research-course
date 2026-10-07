"""Reference solution for lab 17.5."""

from __future__ import annotations

import torch


def magnitude_mask(W, frac):
    k = max(1, int(round(frac * W.numel())))
    thr = W.abs().flatten().kthvalue(W.numel() - k + 1).values
    return W.abs() >= thr


def density_at(step, steps, target):
    return 1 - (1 - target) * min(1.0, step / (steps / 2))


def gated(a, gate, mean):
    return a * gate + mean * (1 - gate)


def rates(grades):
    n = len(grades)
    claims = sum(bool(g["claims"]) for g in grades)
    both = sum(bool(g["claims"]) and bool(g["names"]) for g in grades)
    return {"claims": claims / n, "correct": both / n}
