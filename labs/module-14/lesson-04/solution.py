"""Reference solution for lab 14.4."""

from __future__ import annotations

import numpy as np


def sigmoid_curve(C, A, B, C_mid, R0):
    C = np.asarray(C, dtype=float)
    return R0 + (A - R0) / (1.0 + (C_mid / C) ** B)


def best_asymptote(C, y, B, C_mid, R0, A_max=1.0):
    C, y = np.asarray(C, float), np.asarray(y, float)
    f = 1.0 / (1.0 + (C_mid / C) ** B)
    A = min(R0 + float(f @ (y - R0)) / float(f @ f), A_max)
    r = y - (R0 + (A - R0) * f)
    return A, float(r @ r)


def _pass(n, c, k):
    if n - c < k:
        return 1.0
    p = 1.0
    for i in range(k):
        p *= (n - c - i) / (n - i)
    return 1.0 - p


def pass_at_k_curve(counts, n, ks):
    return np.array([np.mean([_pass(n, int(c), k) for c in counts]) for k in ks])


def crossover(curve_rl, curve_base, ks):
    if not curve_rl[0] > curve_base[0]:
        return None
    for k, a, b in zip(ks, curve_rl, curve_base):
        if b >= a:
            return int(k)
    return None
