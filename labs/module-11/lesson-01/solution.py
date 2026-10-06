"""Reference solution for lab 11.1."""

from __future__ import annotations

import numpy as np


def chinchilla_loss(N, D, c) -> np.ndarray:
    N, D = np.asarray(N, dtype=np.float64), np.asarray(D, dtype=np.float64)
    return c["E"] + c["A"] * N ** -c["alpha"] + c["B"] * D ** -c["beta"]


def compute_optimal(C: float, c, k: float = 6.0) -> tuple[float, float]:
    a, b = c["alpha"], c["beta"]
    G = (a * c["A"] / (b * c["B"])) ** (1.0 / (a + b))
    N = G * (C / k) ** (b / (a + b))
    return float(N), float(C / k / N)


def overtraining_overhead(N: float, D: float, c, k: float = 6.0) -> float:
    L = float(chinchilla_loss(N, D, c))
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        No, Do = compute_optimal(10.0 ** mid, c, k)
        if float(chinchilla_loss(No, Do, c)) > L:
            lo = mid
        else:
            hi = mid
    return k * N * D / 10.0 ** ((lo + hi) / 2) - 1.0


def isoflop_vertex(N, L) -> float:
    c2, c1, _ = np.polyfit(np.log10(np.asarray(N, dtype=np.float64)), np.asarray(L, dtype=np.float64), 2)
    if c2 <= 0:
        raise ValueError("no minimum: the parabola opens downwards")
    return float(10.0 ** (-c1 / (2 * c2)))


def effective_data(U, D_total, R_star: float = 15.387756) -> np.ndarray:
    U, D_total = np.asarray(U, dtype=np.float64), np.asarray(D_total, dtype=np.float64)
    R = np.maximum(D_total / U - 1.0, 0.0)
    return np.where(D_total <= U, D_total, U + U * R_star * (1.0 - np.exp(-R / R_star)))
