"""Reference solution for lab 17.2."""

from __future__ import annotations

import torch


def normalised_effect(m_patched, m_clean, m_corrupt, mode):
    gap = m_clean.mean() - m_corrupt.mean()
    if mode == "denoise":
        return (m_patched - m_corrupt) / gap
    return (m_clean - m_patched) / gap


def head_patch_edit(source_z, head, head_dim):
    sl = slice(head * head_dim, (head + 1) * head_dim)

    def f(x):
        y = x.clone()
        y[..., sl] = source_z[..., sl].to(y.dtype)
        return y
    return f


def head_delta(z_from, z_to, W_O, heads, head_dim):
    mask = torch.zeros(z_to.shape[-1], dtype=z_to.dtype)
    for h in heads:
        mask[h * head_dim:(h + 1) * head_dim] = 1
    return ((z_to - z_from) * mask) @ W_O.T.to(z_to.dtype)


def p_random(effect, random_effects):
    r = [float(x) for x in random_effects]
    return (1 + sum(x >= effect for x in r)) / (1 + len(r))
