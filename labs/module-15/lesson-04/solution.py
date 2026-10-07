"""Reference solution for lab 15.4."""

from __future__ import annotations

import torch


def asym_qdq(x, bits, dim, group):
    xd = x.movedim(dim, -1)
    shp = xd.shape
    g = xd.reshape(*shp[:-1], shp[-1] // group, group)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    L = 2 ** bits - 1
    s = (hi - lo) / L
    safe = torch.where(s > 0, s, torch.ones_like(s))
    q = torch.clamp(torch.round((g - lo) / safe), 0, L)
    return torch.where(s > 0, q * s + lo, g).reshape(shp).movedim(-1, dim)


def kivi_qdq(k, v, bits, group, residual):
    S, d = k.shape[2], k.shape[3]
    n_q = max(0, (S - residual) // group) * group
    if n_q == 0:
        return k, v
    gd = group if d % group == 0 else d
    kq = asym_qdq(k[:, :, :n_q], bits, 2, group)
    vq = asym_qdq(v[:, :, :n_q], bits, 3, gd)
    return torch.cat([kq, k[:, :, n_q:]], 2), torch.cat([vq, v[:, :, n_q:]], 2)


def kv_cache_bytes(S, elements_per_token, bits, group, residual, full_bits=16):
    n_q = max(0, (S - residual) // group) * group
    return (n_q * elements_per_token * (bits + 32 / group) + (S - n_q) * elements_per_token * full_bits) / 8
