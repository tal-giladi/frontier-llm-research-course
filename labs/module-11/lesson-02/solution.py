"""Reference solution for lab 11.2."""

from __future__ import annotations

import numpy as np
import torch


def choice_metrics(lp: torch.Tensor, answer: torch.Tensor, cont: int) -> dict:
    lp = lp.double()
    p = torch.softmax(lp, dim=1)
    return {"acc": float((lp.argmax(1) == answer).double().mean()),
            "p_correct": float(p.gather(1, answer[:, None]).mean()),
            "nll_correct": float((-lp.gather(1, answer[:, None]) / cont).mean())}


def sigmoid_curve(x, x0: float, s: float, lo: float, hi: float) -> np.ndarray:
    return lo + (hi - lo) / (1.0 + np.exp(s * (np.asarray(x, dtype=np.float64) - x0)))


def exact_match_from_token_acc(p, k: int) -> np.ndarray:
    return np.asarray(p, dtype=np.float64) ** k


def explained_variance(scores: np.ndarray) -> np.ndarray:
    X = np.asarray(scores, dtype=np.float64)
    Z = (X - X.mean(0)) / X.std(0, ddof=0)
    s = np.linalg.svd(Z, compute_uv=False)
    return s ** 2 / (s ** 2).sum()


def two_step(nll_target: float, sigmoid_params: dict) -> float:
    p = sigmoid_params
    return float(sigmoid_curve(nll_target, p["x0"], p["s"], p["lo"], p["hi"]))
