"""Reference solution for lab 08.4."""

from __future__ import annotations

import math

import numpy as np


def n_eff(N, P, gamma):
    N = np.asarray(N, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    return N * np.where(np.isinf(P), 1.0, 1.0 - np.exp(-P / gamma))


def p_star(gamma: float, k: int = 3) -> float:
    lo, hi = 1e-6, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if k * mid * math.exp(-mid) - (1 - math.exp(-mid)) > 0:
            lo = mid
        else:
            hi = mid
    return gamma * (lo + hi) / 2


def fit(N, P, L, alphas, gammas) -> dict:
    L = np.asarray(L, dtype=np.float64)
    best = None
    for a in alphas:
        for g in gammas:
            X = np.stack([n_eff(N, P, g) ** -a, np.ones_like(L)], 1)
            coef, *_ = np.linalg.lstsq(X, L, rcond=None)
            if coef[0] <= 0:
                continue
            sse = float(((L - X @ coef) ** 2).sum())
            if best is None or sse < best["sse"]:
                best = {"A": float(coef[0]), "alpha": float(a), "gamma": float(g), "E": float(coef[1]), "sse": sse}
    return best


def ptq_exponent(D, delta) -> float:
    return float(np.polyfit(np.log(np.asarray(D, dtype=np.float64)), np.log(np.asarray(delta, dtype=np.float64)), 1)[0])
