"""Reference solution for lab 17.3."""

from __future__ import annotations

import numpy as np
import torch


def normalise_rows(A):
    Ah = np.abs(A)
    rs = Ah.sum(1, keepdims=True)
    return np.divide(Ah, rs, out=np.zeros_like(Ah, dtype=float), where=rs > 0)


def total_influence(A_hat):
    n = A_hat.shape[0]
    return np.linalg.inv(np.eye(n) - A_hat) - np.eye(n)


def prune_nodes(influence_on_logits, is_logit, threshold=0.8):
    inf = np.asarray(influence_on_logits, dtype=float)
    total = inf.sum()
    keep, run = [], 0.0
    for i in np.argsort(-inf):
        if is_logit[i]:
            continue
        if run / max(total, 1e-12) >= threshold:
            break
        keep.append(int(i))
        run += inf[i]
    return sorted(keep + [int(i) for i in np.nonzero(is_logit)[0]])


def feature_write(W_dec, out_scale, feature, activation):
    return activation * W_dec[feature] * out_scale
