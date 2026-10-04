"""Lab 08.4 (extension) — scaling laws for precision. Fill in the TODOs; run `pytest labs/module-08/lesson-04`.

NumPy only; the tests compare against frontierlab.precision.scaling_law.
"""

from __future__ import annotations

import math  # noqa: F401

import numpy as np  # noqa: F401


def n_eff(N, P, gamma):
    """Effective parameters of weights trained at P bits (arXiv 2411.04330 Eq. 3): N · (1 - exp(-P / γ)).
    ``P`` may contain ``np.inf`` (unquantised: factor 1). Works elementwise on arrays."""
    raise NotImplementedError("TODO 1: effective parameter count")


def p_star(gamma: float, k: int = 3) -> float:
    """Compute-optimal precision of Eq. 5 with α = β: P* = γ · u*, where u* > 0 solves
    k · u · e^(-u) = 1 - e^(-u). Find u* by bisection on (1e-6, 50); the left side is larger below the root."""
    raise NotImplementedError("TODO 2: bisection for u*")


def fit(N, P, L, alphas, gammas) -> dict:
    """Fit L = A · n_eff(N, P, γ)^(-α) + E by trying every (α, γ) in the two grids and solving A and E by least
    squares (np.linalg.lstsq on columns [n_eff^-α, 1]). Skip fits with A <= 0. Return the best as
    {"A", "alpha", "gamma", "E", "sse"} (sse = sum of squared residuals)."""
    raise NotImplementedError("TODO 3: grid + least-squares fit")


def ptq_exponent(D, delta) -> float:
    """The exponent p of delta ∝ D^p (the γ_D of Eq. 2): slope of a straight-line fit of log(delta) on log(D)."""
    raise NotImplementedError("TODO 4: log-log slope")
