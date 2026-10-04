"""Scaling laws for precision (Kumar et al., arXiv 2411.04330): functional forms and a small fitter (lesson 08.4).

Training in low precision acts like having fewer parameters. With weights trained at P_w bits (Eq. 3):

    L(N, D, P_w) = A · N_eff^(-α) + B · D^(-β) + E,      N_eff = N · (1 - exp(-P_w / γ_w))

and with weights, activations and KV cache all quantised (Eq. 4) N_eff multiplies one such factor per tensor.
Post-training quantisation to P_post bits adds (Eq. 2)

    δ_PTQ(N, D, P_post) = C_T · D^γ_D / N^γ_N · exp(-P_post / γ_post)

which *grows* with the training tokens D: the longer a model is trained, the more its weights are hurt by
rounding them afterwards (the paper's abstract: it can eventually make additional pretraining data harmful).

Compute-optimal precision (section 4.3.2, Eq. 5): with training cost C ∝ N · D · P (in units of 16-bit FLOPs,
C = (6/16) · N · D · P), minimise L over N, D and P together. The paper finds P* "around 7-8 bits when fitting our
scaling law on runs with quantization done to integer type" (its fits; it also warns that its "numerical
constants are unlikely to be useful" outside its setting).

:data:`PAPER_CONSTANTS` are the fitted values of the paper's appendix K, Table 2 (α = β tied). The table also has
shift terms (n_w, n_i, n_kv, b) that this module does not use. Plugging γ_w = 2.67 into Eq. 5 gives P* ≈ 5.1
bits, not 7–8: the paper's figure comes from its own fits, which the published constants alone do not let us
recompute (lesson 08.4 shows how P* depends on γ: P* ≈ 1.9·γ).
"""

from __future__ import annotations

import math

import numpy as np

PAPER_CONSTANTS = {"A": 4.299e3, "alpha": 0.4965, "B": 1.806e4, "beta": 0.4965, "E": 2.7648,
                   "gamma_w": 2.6745, "gamma_a": 2.2102, "gamma_kv": 0.9578,
                   "C_T": 0.0598, "gamma_D": 0.5068, "gamma_N": 0.3439, "gamma_post": 0.5907}


def precision_factor(P, gamma) -> np.ndarray:
    """1 - exp(-P / γ): the fraction of parameters that 'count' at P bits."""
    return 1.0 - np.exp(-np.asarray(P, dtype=np.float64) / gamma)


def n_eff(N, P_w=None, gamma_w=2.6745, P_a=None, gamma_a=2.2102, P_kv=None, gamma_kv=0.9578) -> np.ndarray:
    """Eq. 4: N times one factor per quantised tensor type (None = not quantised)."""
    out = np.asarray(N, dtype=np.float64)
    for P, g in ((P_w, gamma_w), (P_a, gamma_a), (P_kv, gamma_kv)):
        if P is not None:
            out = out * precision_factor(P, g)
    return out


def loss(N, D, P_w=None, c=PAPER_CONSTANTS, P_a=None, P_kv=None) -> np.ndarray:
    """Eqs. 3–4: A · N_eff^-α + B · D^-β + E."""
    ne = n_eff(N, P_w, c["gamma_w"], P_a, c["gamma_a"], P_kv, c["gamma_kv"])
    return c["A"] * ne ** -c["alpha"] + c["B"] * np.asarray(D, dtype=np.float64) ** -c["beta"] + c["E"]


def delta_ptq(N, D, P_post, c=PAPER_CONSTANTS) -> np.ndarray:
    """Eq. 2: the loss added by rounding a high-precision-trained model to P_post bits afterwards."""
    N, D = np.asarray(N, dtype=np.float64), np.asarray(D, dtype=np.float64)
    return c["C_T"] * D ** c["gamma_D"] / N ** c["gamma_N"] * np.exp(-np.asarray(P_post, dtype=np.float64) / c["gamma_post"])


def optimal_precision(C: float, gamma: float = PAPER_CONSTANTS["gamma_w"], c=PAPER_CONSTANTS, k: int = 3,
                      P_grid=None, logN_grid=None) -> dict:
    """Eq. 5: minimise A·[N·(1 - e^(-P/γ))^k]^-α + B·D^-β + E subject to C = (6/16)·N·D·P (k = 3: weights,
    activations and KV cache at the same P with one γ, as Eq. 5 writes it).

    For every P on the grid, D follows from C and N, and N is chosen to minimise the loss; returns the best P,
    N and D and the whole curve L*(P). Grid search, exact to the grid spacing. With α = β (the paper ties them)
    the optimum does not depend on C; see :func:`p_star_closed_form`.
    """
    P_grid = np.arange(2.0, 20.01, 0.25) if P_grid is None else np.asarray(P_grid, dtype=np.float64)
    logN = np.linspace(6, 14, 1601) if logN_grid is None else np.asarray(logN_grid)
    N = 10.0 ** logN
    curve = []
    for P in P_grid:
        D = C / ((6 / 16) * N * P)
        L = c["A"] * (N * precision_factor(P, gamma) ** k) ** -c["alpha"] + c["B"] * D ** -c["beta"] + c["E"]
        i = int(np.argmin(L))
        curve.append((float(P), float(L[i]), float(N[i]), float(D[i])))
    best = min(curve, key=lambda t: t[1])
    return {"P_star": best[0], "loss": best[1], "N": best[2], "D": best[3], "curve": curve}


def p_star_closed_form(gamma: float, k: int = 3) -> float:
    """With α = β the compute-optimal loss depends on N_eff·D ∝ C·f(P)/P, f = (1 - e^(-P/γ))^k, so P* maximises
    f(P)/P: k·u·e^(-u) = 1 - e^(-u) with u = P/γ. Solved by bisection; for k = 3, u* ≈ 1.904, so P* ≈ 1.9·γ."""
    lo, hi = 1e-6, 50.0
    h = lambda u: k * u * math.exp(-u) - (1 - math.exp(-u))       # > 0 below the root, < 0 above it
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if h(mid) > 0 else (lo, mid)
    return gamma * (lo + hi) / 2


def fit_precision_law(N, P, L, alphas=None, gammas=None) -> dict:
    """Fit L = A · (N · (1 - exp(-P/γ)))^-α + E' at fixed D, by grid search over (α, γ) and least squares for (A, E').

    ``P`` may contain ``math.inf`` (or None) for unquantised runs (factor 1). Returns the best (A, α, γ, E') and the
    residual RMS. With a handful of points this is a curve fit, not a test of the law.
    """
    N = np.asarray(N, dtype=np.float64)
    P = np.asarray([math.inf if p is None else p for p in P], dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    alphas = np.linspace(0.05, 1.5, 59) if alphas is None else np.asarray(alphas)
    gammas = np.linspace(0.2, 6.0, 117) if gammas is None else np.asarray(gammas)
    best = None
    for a in alphas:
        for g in gammas:
            ne = N * np.where(np.isinf(P), 1.0, 1.0 - np.exp(-P / g))
            X = np.stack([ne ** -a, np.ones_like(ne)], 1)
            coef, *_ = np.linalg.lstsq(X, L, rcond=None)
            if coef[0] <= 0:
                continue
            r = L - X @ coef
            sse = float(r @ r)
            if best is None or sse < best["sse"]:
                best = {"A": float(coef[0]), "alpha": float(a), "gamma": float(g), "E": float(coef[1]), "sse": sse}
    best["rms"] = math.sqrt(best["sse"] / len(L))
    return best


def fit_power(x, y) -> dict:
    """Least-squares fit of log y = log k + p · log x (for δ_PTQ ∝ D^γ_D). Returns k, p and the log-residual RMS."""
    lx, ly = np.log(np.asarray(x, dtype=np.float64)), np.log(np.asarray(y, dtype=np.float64))
    p, lk = np.polyfit(lx, ly, 1)
    r = ly - (lk + p * lx)
    return {"k": float(math.exp(lk)), "p": float(p), "log_rms": float(np.sqrt(np.mean(r ** 2)))}
