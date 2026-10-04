"""Reference solution for lab 07.2."""

from __future__ import annotations

import math

import torch


def orthogonal_update_rms(shape):
    # min(A, B) singular values of 1: squared Frobenius norm = min(A, B); divide by A·B entries
    A, B = shape
    return math.sqrt(min(A, B) / (A * B))           # = 1 / sqrt(max(A, B))


def rms_matched_scale(shape, target=0.2):
    return target / orthogonal_update_rms(shape)


def head_max_logits(q, k, scale, mask):
    lg = (q @ k.transpose(-1, -2)) * scale                     # (B, H, T, T)
    return lg.masked_fill(~mask, float("-inf")).amax(dim=(0, 2, 3))


def qk_clip_gamma(smax, tau):
    return torch.clamp(tau / smax, max=1.0)


def clip_gqa_weights(wq, wk, gamma, H, KV, d):
    for h, g in enumerate(gamma.tolist()):
        if g >= 1.0:
            continue
        if KV == H:
            wq[h * d:(h + 1) * d].mul_(math.sqrt(g))
            wk[h * d:(h + 1) * d].mul_(math.sqrt(g))
        else:
            wq[h * d:(h + 1) * d].mul_(g)


def clip_mla_weights(wq, wkv, gamma, H, d_n, d_r, d_v):
    dq, dkv = d_n + d_r, d_n + d_v
    for h, g in enumerate(gamma.tolist()):
        if g >= 1.0:
            continue
        wq[h * dq:h * dq + d_n].mul_(math.sqrt(g))
        wq[h * dq + d_n:(h + 1) * dq].mul_(g)
        wkv[h * dkv:h * dkv + d_n].mul_(math.sqrt(g))
