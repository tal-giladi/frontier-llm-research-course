"""Mixture regression from many tiny runs: RegMix and the data mixing law (lesson 10.4).

RegMix (Liu et al. 2024, section 3) trains many small proxy models on randomly sampled mixtures,
fits a regression from mixture weights to a target loss, and picks the mixture the regression
predicts best. The paper trains 512 models of 1M parameters on 1B tokens each, samples mixtures
from a Dirichlet whose concentration is the token distribution of the sources times a factor drawn
from 0.1–5.0, fits ridge regression and LightGBM, and averages the top 100 predicted mixtures.

The data mixing law (Ye et al. 2024, Eq. 7) is a parametric alternative for the loss on validation
domain i as a function of the training proportions r_1..r_M:

    L_i(r) = c_i + k_i · exp( Σ_j t_ij r_j )

and for a validation set that mixes domains with weights s_i (Eq. 8): L = Σ_i s_i L_i(r).

This module has the three fits used in the lab, all NumPy / PyTorch float64:

* :func:`fit_ridge` — linear in r (RegMix's linear model), closed form;
* :func:`fit_ridge` with ``quadratic=True`` — adds pairwise products r_j r_k, a cheap non-linear model
  (the course does not install LightGBM; INFERENCE: any smooth non-linear regressor plays the same role);
* :func:`fit_mixing_law` — Eq. 7 for one target, by Adam then L-BFGS on the squared error;

plus :func:`sample_mixtures` (RegMix's Dirichlet sampling), :func:`loo_predictions` (leave-one-out),
:func:`rank_corr` (Spearman, the paper's metric) and :func:`best_mixture` (search the simplex).
"""

from __future__ import annotations

import numpy as np
import torch

from frontierlab.datax.quality import spearman


def sample_mixtures(n: int, prior, seed: int = 0, lo: float = 0.1, hi: float = 5.0) -> np.ndarray:
    """(n, M) mixtures: each row ~ Dirichlet(alpha = prior · u), u ~ Uniform(lo, hi) per row (RegMix 3.1)."""
    rng = np.random.default_rng(seed)
    prior = np.asarray(prior, dtype=np.float64)
    prior = prior / prior.sum()
    rows = []
    for _ in range(n):
        alpha = np.maximum(prior * rng.uniform(lo, hi), 1e-3)
        rows.append(rng.dirichlet(alpha))
    return np.asarray(rows)


def _design(X: np.ndarray, quadratic: bool) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    cols = [np.ones((X.shape[0], 1)), X]
    if quadratic:
        M = X.shape[1]
        cols.append(np.stack([X[:, j] * X[:, k] for j in range(M) for k in range(j, M)], axis=1))
    return np.concatenate(cols, axis=1)


def fit_ridge(X, y, l2: float = 1e-4, quadratic: bool = False) -> dict:
    """Ridge regression of y on r (and r_j r_k if quadratic); the intercept is not penalised."""
    A = _design(X, quadratic)
    P = l2 * np.eye(A.shape[1])
    P[0, 0] = 0.0
    w = np.linalg.solve(A.T @ A + P, A.T @ np.asarray(y, dtype=np.float64))
    return {"kind": "ridge2" if quadratic else "ridge", "w": w, "quadratic": quadratic}


def fit_mixing_law(X, y, steps: int = 3000, seed: int = 0) -> dict:
    """Fit L(r) = c + k·exp(r·t) (Eq. 7) by least squares; k > 0 via k = exp(log_k)."""
    X = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(y), dtype=torch.float64)
    torch.manual_seed(seed)
    c = torch.tensor(float(y.min()) - 0.1 * float(y.std() + 1e-3), dtype=torch.float64, requires_grad=True)
    log_k = torch.tensor(float(np.log(max(1e-3, float(y.mean() - c.detach())))), dtype=torch.float64, requires_grad=True)
    t = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    params = [c, log_k, t]

    def loss():
        return ((c + torch.exp(log_k + X @ t) - y) ** 2).mean()

    opt = torch.optim.Adam(params, lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        lo = loss()
        lo.backward()
        opt.step()
    lb = torch.optim.LBFGS(params, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        lb.zero_grad()
        lo = loss()
        lo.backward()
        return lo
    lb.step(closure)
    return {"kind": "law", "c": float(c.detach()), "k": float(torch.exp(log_k.detach())), "t": t.detach().numpy().copy(),
            "mse": float(loss())}


def predict(fit: dict, X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if fit["kind"] == "law":
        return fit["c"] + fit["k"] * np.exp(X @ fit["t"])
    return _design(X, fit["quadratic"]) @ fit["w"]


def fit_any(kind: str, X, y, **kw) -> dict:
    if kind == "ridge":
        return fit_ridge(X, y, quadratic=False, **kw)
    if kind == "ridge2":
        return fit_ridge(X, y, quadratic=True, **kw)
    if kind == "law":
        return fit_mixing_law(X, y, **kw)
    raise KeyError(kind)


def loo_predictions(kind: str, X, y, **kw) -> np.ndarray:
    """Leave-one-out predictions: run i predicted by a fit on all other runs."""
    X, y = np.asarray(X), np.asarray(y)
    out = np.empty(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        out[i] = predict(fit_any(kind, X[keep], y[keep], **kw), X[i:i + 1])[0]
    return out


def rank_corr(pred, y) -> float:
    return spearman(pred, y)


def best_mixture(fit: dict, M: int, prior=None, n: int = 100000, top: int = 100, seed: int = 1) -> dict:
    """Search: sample ``n`` mixtures (RegMix sampling around ``prior``), predict, average the ``top`` best."""
    prior = np.ones(M) / M if prior is None else np.asarray(prior, dtype=np.float64)
    cand = sample_mixtures(n, prior, seed)
    pred = predict(fit, cand)
    order = np.argsort(pred)
    best = cand[order[:top]].mean(axis=0)
    return {"mixture": best, "predicted": float(predict(fit, best[None])[0]),
            "best_single": cand[order[0]], "best_single_predicted": float(pred[order[0]])}
