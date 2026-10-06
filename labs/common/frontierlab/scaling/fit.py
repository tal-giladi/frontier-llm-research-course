"""Fitting scaling laws to a ladder of runs (lessons 11.1 and 11.3).

Three fits, the three ways Hoffmann et al. (arXiv 2203.15556, section 3) estimate compute-optimal scaling,
written for a handful of runs:

* :func:`isoflop_minima` (their approach 2): at each compute budget, fit a parabola of loss against log N and
  take its minimum; then :func:`fit_allocation` fits ``N_opt ∝ C^a`` through the minima.
* :func:`fit_parametric` (their approach 3): fit ``L = E + A·N^-α + B·D^-β`` to every run at once, minimising
  a Huber loss on log-loss residuals (their Eq. 3). Here: a grid over (α, β) with non-negative linear least
  squares for (E, A, B) at each grid point, then an L-BFGS refinement of all five in float64.
* :func:`fit_power_offset`: ``L(C) = E + k·C^-γ`` through the best run per budget (the GPT-4 report's form for
  predicting final loss from compute, arXiv 2303.08774 section 3.1).

:func:`bootstrap` refits on runs resampled with replacement, which gives an interval for any prediction;
:func:`holdout` fits on part of the ladder and reports the error on the rest. A good in-sample fit is not
evidence that the law extrapolates; only held-out points are.
"""

from __future__ import annotations

import itertools
import math

import numpy as np


def _nnls3(X: np.ndarray, y: np.ndarray):
    """Least squares with coefficients >= 0 for a design with at most 3 columns (exact: try every support)."""
    best, best_sse = None, math.inf
    m = X.shape[1]
    for r in range(1, m + 1):
        for cols in itertools.combinations(range(m), r):
            coef, *_ = np.linalg.lstsq(X[:, cols], y, rcond=None)
            if (coef < 0).any():
                continue
            full = np.zeros(m)
            full[list(cols)] = coef
            res = y - X @ full
            sse = float(res @ res)
            if sse < best_sse:
                best, best_sse = full, sse
    return best, best_sse


def huber(r: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r ** 2, delta * (a - 0.5 * delta))


def predict(fit: dict, N, D) -> np.ndarray:
    N, D = np.asarray(N, dtype=np.float64), np.asarray(D, dtype=np.float64)
    return fit["E"] + fit["A"] * N ** -fit["alpha"] + fit["B"] * D ** -fit["beta"]


def fit_parametric(N, D, L, alphas=None, betas=None, delta: float = 1e-3, refine: bool = True,
                   fix: dict | None = None) -> dict:
    """Fit L = E + A·N^-α + B·D^-β to runs (N_i, D_i, L_i).

    Grid search over (α, β) with non-negative least squares for (E, A, B) on the losses, then (``refine``) an
    L-BFGS minimisation of Σ Huber_δ(log L̂_i − log L_i) over (log A, log B, log E, α, β) from the best grid
    point, in float64. ``fix`` holds exponents to keep fixed, e.g. ``{"alpha": 0.34, "beta": 0.28}`` (then only
    E, A, B are fitted, by linear least squares: lesson 11.3 refits at intermediate points this way).
    Returns E, A, B, alpha, beta, the in-sample RMS of log residuals and the grid point it started from.
    """
    N, D, L = (np.asarray(v, dtype=np.float64) for v in (N, D, L))
    fix = fix or {}
    alphas = np.array([fix["alpha"]]) if "alpha" in fix else (np.linspace(0.05, 1.2, 47) if alphas is None else np.asarray(alphas))
    betas = np.array([fix["beta"]]) if "beta" in fix else (np.linspace(0.05, 1.2, 47) if betas is None else np.asarray(betas))
    best = None
    for a in alphas:
        for b in betas:
            X = np.stack([np.ones_like(N), N ** -a, D ** -b], 1)
            coef, sse = _nnls3(X, L)
            if coef is None:
                continue
            if best is None or sse < best["sse"]:
                best = {"E": coef[0], "A": coef[1], "B": coef[2], "alpha": float(a), "beta": float(b), "sse": sse}
    if best is None:
        raise ValueError("no non-negative fit found")
    grid = {k: float(best[k]) for k in ("E", "A", "B", "alpha", "beta")}
    out = dict(grid)
    if refine and not fix and min(grid["E"], grid["A"], grid["B"]) > 0:
        out = _refine(N, D, L, grid, delta)
    out["rms_log"] = float(np.sqrt(np.mean((np.log(predict(out, N, D)) - np.log(L)) ** 2)))
    out["grid"] = grid
    out["n"] = int(N.size)
    return out


def _refine(N, D, L, start: dict, delta: float) -> dict:
    import torch
    t = lambda v: torch.tensor(v, dtype=torch.float64)
    lN, lD, lL = t(np.log(N)), t(np.log(D)), t(np.log(L))
    p = torch.tensor([math.log(start["A"]), math.log(start["B"]), math.log(start["E"]), start["alpha"], start["beta"]],
                     dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([p], lr=1.0, max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-14,
                            line_search_fn="strong_wolfe")

    def objective():
        a, b, e, al, be = p
        pred = torch.logsumexp(torch.stack([a - al * lN, b - be * lD, e.expand_as(lN)]), 0)   # Hoffmann et al. Eq. 3
        r = pred - lL
        return torch.where(r.abs() <= delta, 0.5 * r ** 2, delta * (r.abs() - 0.5 * delta)).sum()

    def closure():
        opt.zero_grad()
        f = objective()
        f.backward()
        return f

    with torch.no_grad():
        f0 = float(objective())
    opt.step(closure)
    a, b, e, al, be = p.detach().tolist()
    cand = {"E": math.exp(e), "A": math.exp(a), "B": math.exp(b), "alpha": al, "beta": be}
    with torch.no_grad():
        f1 = float(objective())
    if not all(math.isfinite(v) for v in cand.values()) or f1 > f0 or al <= 0 or be <= 0:
        return dict(start)
    return cand


def isoflop_minima(runs: list[dict], key_N: str = "N") -> list[dict]:
    """Approach 2: per budget, a parabola L = c0 + c1·x + c2·x² in x = log10 N; its vertex is N_opt.

    ``runs``: dicts with ``budget`` (the nominal C of the iso-FLOP group), ``C`` (the run's actual FLOPs),
    ``key_N``, ``D`` and ``loss``. ``D_opt`` is interpolated in log space between the runs' token counts (so it
    respects the real FLOPs per token, not 6·N·D). ``edge`` is True when the vertex lies outside the sampled sizes
    (an extrapolated minimum: widen the grid before trusting it).
    """
    out = []
    for budget in sorted({r["budget"] for r in runs}):
        g = sorted((r for r in runs if r["budget"] == budget), key=lambda r: r[key_N])
        if len(g) < 3:
            continue
        x = np.log10([r[key_N] for r in g])
        y = np.array([r["loss"] for r in g])
        c2, c1, c0 = np.polyfit(x, y, 2)
        if c2 <= 0:
            out.append({"budget": budget, "N_opt": math.nan, "loss_min": float(y.min()), "edge": True, "curvature": float(c2)})
            continue
        xv = -c1 / (2 * c2)
        lD = np.log10([r["D"] for r in g])
        D_opt = 10.0 ** float(np.interp(xv, x, lD)) if x[0] <= xv <= x[-1] else 10.0 ** float(np.polyval(np.polyfit(x, lD, 1), xv))
        out.append({"budget": budget, "N_opt": float(10.0 ** xv), "D_opt": D_opt, "loss_min": float(c0 + c1 * xv + c2 * xv ** 2),
                    "edge": not (x[0] <= xv <= x[-1]), "curvature": float(c2)})
    return out


def fit_allocation(C, N_opt) -> dict:
    """log10 N_opt = a·log10 C + b: the exponent a of N_opt ∝ C^a (0.5 means "scale N and D equally")."""
    lc, ln = np.log10(np.asarray(C, dtype=np.float64)), np.log10(np.asarray(N_opt, dtype=np.float64))
    a, b = np.polyfit(lc, ln, 1)
    return {"a": float(a), "log10_k": float(b), "predict": lambda c: 10.0 ** (a * np.log10(c) + b)}


def fit_power_offset(C, L, E_grid=None) -> dict:
    """L(C) = E + k·C^-γ: grid over E below min(L), linear least squares of log(L − E) on log C at each E."""
    C, L = np.asarray(C, dtype=np.float64), np.asarray(L, dtype=np.float64)
    E_grid = np.linspace(0.0, L.min() - 1e-3, 400) if E_grid is None else np.asarray(E_grid)
    best = None
    for E in E_grid:
        y = np.log(L - E)
        g, lk = np.polyfit(np.log(C), y, 1)
        r = y - (lk + g * np.log(C))
        sse = float(r @ r)
        if g < 0 and (best is None or sse < best["sse"]):
            best = {"E": float(E), "k": float(math.exp(lk)), "gamma": float(-g), "sse": sse}
    if best is None:
        raise ValueError("loss does not decrease with compute")
    return best


def predict_power_offset(fit: dict, C) -> np.ndarray:
    return fit["E"] + fit["k"] * np.asarray(C, dtype=np.float64) ** -fit["gamma"]


def bootstrap(N, D, L, n_boot: int = 200, seed: int = 0, **kw) -> list[dict]:
    """Parametric fits on ``n_boot`` resamples of the runs (with replacement). Resamples with fewer than five
    distinct runs are skipped (the law has five parameters)."""
    N, D, L = (np.asarray(v, dtype=np.float64) for v in (N, D, L))
    rng = np.random.default_rng(seed)
    fits = []
    for _ in range(n_boot):
        idx = rng.integers(0, N.size, N.size)
        if len(set(idx.tolist())) < 5:
            continue
        try:
            fits.append(fit_parametric(N[idx], D[idx], L[idx], refine=False, **kw))
        except ValueError:
            continue
    return fits


def prediction_interval(fits: list[dict], N, D, level: float = 0.9) -> tuple[float, float, float]:
    """Median and central ``level`` interval of the bootstrap fits' predictions at (N, D)."""
    p = np.array([float(predict(f, N, D)) for f in fits])
    lo, mid, hi = np.quantile(p, [(1 - level) / 2, 0.5, (1 + level) / 2])
    return float(mid), float(lo), float(hi)


def holdout(N, D, L, test_mask, **kw) -> dict:
    """Fit on the runs where ``test_mask`` is False; report predictions and errors on the others."""
    N, D, L = (np.asarray(v, dtype=np.float64) for v in (N, D, L))
    m = np.asarray(test_mask, dtype=bool)
    fit = fit_parametric(N[~m], D[~m], L[~m], **kw)
    pred = predict(fit, N[m], D[m])
    err = pred - L[m]
    return {"fit": fit, "pred": pred.tolist(), "measured": L[m].tolist(), "err": err.tolist(),
            "mae": float(np.abs(err).mean()), "max_abs": float(np.abs(err).max())}
