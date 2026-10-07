"""Reference solution for lab 17.4."""

from __future__ import annotations

import numpy as np
import torch


def mean_diff(pos_acts, neg_acts):
    return pos_acts.float().mean(0) - neg_acts.float().mean(0)


def steer_edit(vec, alpha, start=0):
    def f(x):
        y = x.clone()
        y[:, start:] = y[:, start:] + (alpha * vec).to(y.dtype)
        return y
    return f


def token_kl(logp_ref, logp_new):
    return (logp_ref.exp() * (logp_ref - logp_new)).sum(-1)


def flip_rate(base_logodds, steered_logodds):
    b, s = np.asarray(base_logodds), np.asarray(steered_logodds)
    return float(((b > 0) != (s > 0)).mean())
