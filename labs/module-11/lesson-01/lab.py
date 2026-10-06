"""Lab 11.1 — compute-optimal and over-trained regimes. Fill in the TODOs; run `pytest labs/module-11/lesson-01`.

All inputs are floats or NumPy arrays in float64. ``c`` is a dict with keys E, A, B, alpha, beta for the law
L(N, D) = E + A·N^-alpha + B·D^-beta (frontierlab.scaling.laws.CHINCHILLA is Hoffmann et al.'s fit).
"""

from __future__ import annotations

import numpy as np


def chinchilla_loss(N, D, c) -> np.ndarray:
    """L(N, D) = E + A·N^-alpha + B·D^-beta."""
    raise NotImplementedError("TODO 1: the parametric loss")


def compute_optimal(C: float, c, k: float = 6.0) -> tuple[float, float]:
    """(N_opt, D_opt) minimising L(N, D) subject to C = k·N·D, in closed form:
    G = (alpha·A / (beta·B))^(1/(alpha+beta)); N_opt = G·(C/k)^(beta/(alpha+beta)); D_opt = (C/k) / N_opt."""
    raise NotImplementedError("TODO 2: the compute-optimal allocation")


def overtraining_overhead(N: float, D: float, c, k: float = 6.0) -> float:
    """Training compute spent by (N, D) divided by the compute a compute-optimal run needs to reach the same loss,
    minus 1. Find that compute by bisection on log10 C (between 0 and 40) using your compute_optimal: the
    compute-optimal loss decreases with C."""
    raise NotImplementedError("TODO 3: the price of over-training")


def isoflop_vertex(N, L) -> float:
    """Approach 2 at one budget: fit L = c0 + c1·x + c2·x² with x = log10 N (np.polyfit, degree 2) and return
    the N at the vertex, 10^(-c1 / (2·c2)). Raise ValueError if c2 <= 0 (no minimum)."""
    raise NotImplementedError("TODO 4: the iso-FLOP minimum")


def effective_data(U, D_total, R_star: float = 15.387756) -> np.ndarray:
    """Muennighoff et al. Eq. 5: D' = U + U·R*·(1 − exp(−R/R*)) with R = D_total/U − 1 repeats; D' = D_total when
    D_total <= U (no repetition)."""
    raise NotImplementedError("TODO 5: the value of repeated tokens")
