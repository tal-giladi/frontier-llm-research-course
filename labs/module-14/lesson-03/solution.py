"""Reference solution for lab 14.3."""

from __future__ import annotations

import numpy as np
import torch


def simulate(gen_times, train_time, bound):
    g = np.asarray(gen_times, dtype=float)
    n = g.size
    upd_end = np.zeros(n)
    lags = []
    prev_gen_end = 0.0
    for j in range(n):
        ready = upd_end[j - bound - 1] if j - bound >= 1 else 0.0
        s = max(prev_gen_end, ready)
        version = int((upd_end[:j] <= s).sum())
        lags.append(j - version)
        prev_gen_end = s + g[j]
        upd_end[j] = max(prev_gen_end, upd_end[j - 1] if j else 0.0) + train_time
    return {"total_s": float(upd_end[-1]), "lags": lags, "max_lag": int(max(lags)), "mean_lag": float(np.mean(lags))}


def mismatch_stats(trainer_logp, sampler_logp, mask):
    m = mask.double()
    n = m.sum().clamp_min(1.0)
    d = (trainer_logp.double() - sampler_logp.double()) * m
    on = m.bool()
    return {"mean_abs": float(d.abs().sum() / n), "max_abs": float(d.abs().max()),
            "k3": float(((torch.exp(-d) - 1 + d) * m).sum() / n), "max_ratio": float(torch.exp(d[on]).max()),
            "seq_logratio_absmax": float(d.sum(-1).abs().max())}


def fixed_tile_matmul(x, w, tile=64):
    out = []
    for i in range(0, x.shape[0], tile):
        blk = x[i:i + tile]
        pad = tile - blk.shape[0]
        if pad:
            blk = torch.cat([blk, blk.new_zeros(pad, blk.shape[1])])
        out.append((blk @ w)[: tile - pad])
    return torch.cat(out)
