"""Compute-performance curves for RL (lesson 14.4): ScaleRL's sigmoid, its fit, and how well a short run
identifies the asymptote.

ScaleRL (Khatri et al. 2025, Eq. 1) models the expected pass rate on held-out prompts after RL compute C as

    R_C = R_0 + (A - R_0) / (1 + (C_mid / C)^B)

with R_0 the starting pass rate, A <= 1 the asymptote, B > 0 the scaling exponent and C_mid the compute at
which half of the gain A - R_0 is reached. For fixed (B, C_mid) the curve is linear in A, so :func:`fit_sigmoid`
solves for A in closed form on a grid of (B, log C_mid) and refines the best point; :func:`profile_asymptote`
fixes A on a grid and minimises over (B, C_mid), which shows directly which asymptotes the data rule out.

Reported by ScaleRL (section 2.1 and the appendix fit-window study): fits begin after ~1.5k GPU-hours; the
8B run's fit on (1.5k, 50k) GPU-hours gives A = 0.645, B = 1.70; run-to-run variation of A was within
+-0.015 over 3 runs. C_mid is not given numerically in the text.
"""

from __future__ import annotations

import math

import numpy as np


def sigmoid_curve(C, A: float, B: float, C_mid: float, R0: float) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    return R0 + (A - R0) / (1.0 + (C_mid / C) ** B)


def _best_A(C, y, B, C_mid, R0, A_max=1.0):
    f = 1.0 / (1.0 + (C_mid / C) ** B)
    den = float(f @ f)
    a = float(f @ (y - R0)) / den if den > 0 else 0.0
    A = min(max(R0 + a, 0.0), A_max)          # a pass rate: 0 <= A <= 1
    r = y - (R0 + (A - R0) * f)
    return A, float(r @ r)


def fit_sigmoid(C, y, R0: float, B_grid=None, logcmid_grid=None, A_max: float = 1.0, refine: bool = True) -> dict:
    """Least-squares fit of (A, B, C_mid) with R0 fixed. Returns the parameters and the residual sum of squares."""
    C, y = np.asarray(C, float), np.asarray(y, float)
    B_grid = np.linspace(0.3, 4.0, 38) if B_grid is None else B_grid
    lo, hi = math.log(C.min()) - 2, math.log(C.max()) + 6
    logcmid_grid = np.linspace(lo, hi, 81) if logcmid_grid is None else logcmid_grid
    best = (math.inf, None)
    for B in B_grid:
        for lc in logcmid_grid:
            A, sse = _best_A(C, y, B, math.exp(lc), R0, A_max)
            if sse < best[0]:
                best = (sse, (A, B, lc))
    sse, (A, B, lc) = best
    if refine:
        from scipy.optimize import minimize

        def obj(p):
            Bv, lcv = p
            if Bv <= 0.05:
                return 1e9
            return _best_A(C, y, Bv, math.exp(lcv), R0, A_max)[1]

        res = minimize(obj, [B, lc], method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 4000})
        if res.fun <= sse:
            B, lc = res.x
            A, sse = _best_A(C, y, B, math.exp(lc), R0, A_max)
    return {"A": float(A), "B": float(B), "C_mid": float(math.exp(lc)), "R0": R0, "sse": float(sse), "n": int(C.size)}


def profile_asymptote(C, y, R0: float, A_grid=None, B_grid=None, logcmid_grid=None) -> dict:
    """For each fixed A, the smallest SSE over (B, C_mid). Returns the profile and the 95% profile interval for A.

    The interval uses delta-chi-square <= 3.84 with the noise variance estimated from the best fit's residuals,
    SSE_min / (n - 3). An interval that runs to the edge of the grid (A = 1) means the data cannot bound A.
    """
    C, y = np.asarray(C, float), np.asarray(y, float)
    fit0 = fit_sigmoid(C, y, R0)
    if A_grid is None:                       # coarse everywhere, fine (0.001) near the best fit
        fine = np.arange(fit0["A"] - 0.06, fit0["A"] + 0.06, 0.001)
        A_grid = np.unique(np.concatenate([np.linspace(R0 + 0.005, 1.0, 140), fine[(fine > R0) & (fine <= 1.0)]]))
    A_grid = np.asarray(A_grid, float)
    B_grid = np.linspace(0.3, 4.0, 38) if B_grid is None else B_grid
    lo, hi = math.log(C.min()) - 2, math.log(C.max()) + 8
    logcmid_grid = np.linspace(lo, hi, 101) if logcmid_grid is None else logcmid_grid
    from scipy.optimize import minimize
    cm = np.exp(logcmid_grid)
    prof = []
    for A in A_grid:
        best, arg = math.inf, None
        for B in B_grid:
            f = 1.0 / (1.0 + (cm[:, None] / C[None, :]) ** B)       # (n_cmid, n)
            r = y[None, :] - (R0 + (A - R0) * f)
            sse = (r * r).sum(1)
            j = int(sse.argmin())
            if sse[j] < best:
                best, arg = float(sse[j]), (B, logcmid_grid[j])

        def obj(p, A=A):
            if p[0] <= 0.05:
                return 1e9
            r = y - sigmoid_curve(C, A, p[0], math.exp(p[1]), R0)
            return float(r @ r)

        res = minimize(obj, list(arg), method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-14, "maxiter": 2000})
        prof.append(min(best, float(res.fun)))
    prof = np.asarray(prof)
    fit = fit0
    sse_min = min(fit["sse"], float(prof.min()))
    sigma2 = max(sse_min / max(1, C.size - 3), 1e-12)
    inside = (prof - sse_min) / sigma2 <= 3.84
    idx = np.nonzero(inside)[0]
    lo_A = float(A_grid[idx[0]]) if idx.size else float("nan")
    hi_A = float(A_grid[idx[-1]]) if idx.size else float("nan")
    return {"A_grid": A_grid.tolist(), "sse": prof.tolist(), "fit": fit, "interval": (lo_A, hi_A),
            "bounded_above": bool(idx.size and idx[-1] < len(A_grid) - 1), "sigma": math.sqrt(sigma2)}


def window(C, y, lo: float, hi: float):
    """The points with lo <= C <= hi (a fit window, as in ScaleRL's fit-window study)."""
    C, y = np.asarray(C, float), np.asarray(y, float)
    k = (C >= lo) & (C <= hi)
    return C[k], y[k]
