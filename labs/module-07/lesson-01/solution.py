"""Reference solution for lab 07.1."""

from __future__ import annotations

import math

import torch

QUINTIC = (3.4445, -4.7750, 2.0315)


def ns_step(X, a, b, c):
    A = X @ X.T                         # (r, r)
    B = b * A + c * (A @ A)             # (r, r)
    return a * X + B @ X                # (r, n): each singular value s -> a s + b s^3 + c s^5


def newton_schulz(G, coeffs=QUINTIC, steps=5, eps=1e-7):
    X = G
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        X = ns_step(X, *coeffs)
    return X.T if tall else X


def muon_direction(grad, buf, momentum=0.95, nesterov=True):
    buf.mul_(momentum).add_(grad)
    u = grad + momentum * buf if nesterov else buf
    return newton_schulz(u)


def muon_apply(W, O, lr, weight_decay, adjust="match_rms"):
    A, B = W.shape
    s = 0.2 * math.sqrt(max(A, B)) if adjust == "match_rms" else math.sqrt(max(1.0, A / B))
    W.mul_(1 - lr * weight_decay)
    W.sub_(O, alpha=lr * s)


def muon_param_names(model):
    return [n for n, p in model.named_parameters()
            if p.ndim == 2 and "embed_tokens" not in n and "lm_head" not in n and "norm" not in n]


def ns_flops(shape, steps=5):
    r, n = min(shape), max(shape)
    return steps * (2 * r * r * n + 2 * r ** 3 + 2 * r * r * n)
