"""Lab 12.1 — rewards. Reference solution."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import gammaln


def bt_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def bon_kl(n: int) -> float:
    return math.log(n) - (n - 1) / n


def bon_expected(proxy, value, n: int) -> float:
    proxy, value = np.asarray(proxy, dtype=np.float64), np.asarray(value, dtype=np.float64)
    N = proxy.size
    order = np.argsort(proxy, kind="stable")
    i = np.arange(1, N + 1)
    logw = np.full(N, -np.inf)
    ok = i >= n
    a, b = i[ok] - 1, n - 1
    logw[ok] = (gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)
                - (gammaln(N + 1) - gammaln(n + 1) - gammaln(N - n + 1)))
    return float((np.exp(logw) * value[order]).sum())


def ece(p, y, bins: int = 10) -> float:
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if sel.any():
            total += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(total)


def verifier_errors(accepted, truly_correct) -> dict:
    a, t = np.asarray(accepted, dtype=bool), np.asarray(truly_correct, dtype=bool)
    return {"false_positive_rate": float((a & ~t).sum() / max(1, (~t).sum())),
            "false_negative_rate": float((~a & t).sum() / max(1, t.sum())),
            "precision": float((a & t).sum() / max(1, a.sum()))}


def peak(n_values, gold_values, tol: float = 0.0) -> dict:
    g = np.asarray(gold_values, dtype=np.float64)
    i = int(np.argmax(g))
    return {"n_peak": int(n_values[i]), "gold_peak": float(g[i]), "gold_last": float(g[-1]),
            "declined": bool(g[-1] < g[i] - tol)}
