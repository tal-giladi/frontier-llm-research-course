"""Reference solution for lab 10.4 — mixtures and micro-anneals."""

from __future__ import annotations

import numpy as np


def sample_mixtures(n, prior, seed=0, lo=0.1, hi=5.0):
    rng = np.random.default_rng(seed)
    prior = np.asarray(prior, dtype=np.float64)
    prior = prior / prior.sum()
    rows = []
    for _ in range(n):
        alpha = np.maximum(prior * rng.uniform(lo, hi), 1e-3)
        rows.append(rng.dirichlet(alpha))
    return np.asarray(rows)


def ridge_fit(X, y, l2=1e-4):
    X = np.asarray(X, dtype=np.float64)
    A = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    P = l2 * np.eye(A.shape[1])
    P[0, 0] = 0.0
    return np.linalg.solve(A.T @ A + P, A.T @ np.asarray(y, dtype=np.float64))


def ridge_predict(w, X):
    X = np.asarray(X, dtype=np.float64)
    return w[0] + X @ w[1:]


def law_predict(c, k, t, X):
    return c + k * np.exp(np.asarray(X, dtype=np.float64) @ np.asarray(t, dtype=np.float64))


def loo_ridge(X, y, l2=1e-4):
    X, y = np.asarray(X), np.asarray(y)
    out = np.empty(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        out[i] = ridge_predict(ridge_fit(X[keep], y[keep], l2), X[i:i + 1])[0]
    return out


def anneal_lr(step, steps, lr, warmup):
    if step < warmup:
        return lr * (step + 1) / warmup
    return lr * max(0.0, (steps - step) / max(1, steps - warmup))


def anneal_decision(target_ci, guard_ci, max_regression):
    """target_ci: (candidate − control) loss on the target domain; guard_ci: the same elsewhere."""
    if target_ci[1] < 0 and guard_ci[1] <= max_regression:
        return "adopt"
    if target_ci[0] >= 0 or guard_ci[0] > max_regression:
        return "reject"
    return "inconclusive"
