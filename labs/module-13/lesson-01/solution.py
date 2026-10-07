"""Lab 13.1 — Open recipes as case studies. Reference solution."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def dpo_loss(pol_c, pol_r, ref_c, ref_r, beta, len_c=None, len_r=None, normalise=False, nll_coef=0.0):
    lc, lr = pol_c - ref_c, pol_r - ref_r
    if normalise:
        lc, lr = lc / len_c.clamp_min(1.0), lr / len_r.clamp_min(1.0)
    loss = -F.logsigmoid(beta * (lc - lr)).mean()
    if nll_coef > 0:
        loss = loss + nll_coef * (-(pol_c / len_c.clamp_min(1.0))).mean()
    return loss


def binarise(scores, rng):
    s = np.asarray(scores, dtype=float)
    if s.max() == s.min():
        return None
    top = np.flatnonzero(s == s.max())
    low = np.flatnonzero(s < s.max())
    c = int(rng.choice(top))
    r = int(rng.choice(low))
    return c, r


def binomial_se(p, n):
    return math.sqrt(p * (1 - p) / n)


def mde_unpaired(p, n, z=1.96):
    return z * math.sqrt(2 * p * (1 - p) / n)


def audit(stages, n_items, seed_spread=None):
    rows = []
    for (a, sa), (b, sb) in zip(stages, stages[1:]):
        d = sb - sa
        mde = 100 * mde_unpaired(sa / 100, n_items) if n_items else None
        if mde is not None:
            verdict = "beyond item noise" if abs(d) > mde else "within item noise"
        elif seed_spread is not None:
            verdict = "beyond seed spread" if abs(d) > seed_spread else "within seed spread"
        else:
            verdict = "no noise estimate"
        rows.append({"from": a, "to": b, "delta": d, "mde": mde, "verdict": verdict})
    return rows
