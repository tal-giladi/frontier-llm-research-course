"""A colleague's claim harness. Their message:

    "Head 0.2 of the induction model and our 8 IOI heads in Qwen3-0.6B both pass every check: the effect is huge,
    the held-out effect is just as large, and no random head set comes close. I simplified the harness a bit:
    per-prompt normalisation is more precise than dividing by a batch mean, the held-out set is just a fresh
    sample (new seed, so new prompts), and the random sets are drawn from all heads so the control is fair."

There are three planted bugs. Same API as ``harness.py``.
"""

from __future__ import annotations

import numpy as np
import torch

from frontierlab.interp import tasks as T


def effect(m_patched, m_clean, m_corrupt, mode="noise"):
    gap = m_clean - m_corrupt                                   # per prompt
    return (m_clean - m_patched) / gap if mode == "noise" else (m_patched - m_corrupt) / gap


def heldout(kind, n, seed=0, tok=None):
    if kind == "induction":
        return T.induction_pairs(n, kind="key", seed=seed + 101)          # a new seed, same distribution
    return T.as_pairs(T.ioi_prompts(tok, "train", n, seed + 1, T.NAMES[:12]))


def selection(kind, n, seed=0, tok=None):
    if kind == "induction":
        return T.induction_pairs(n, kind="key", seed=seed)
    return T.as_pairs(T.ioi_prompts(tok, "train", n, seed, T.NAMES[:12]))


def random_sets(candidate, n_layers, n_heads, n, seed=0, layer_matched=True):
    pool = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        l, h = pool[int(rng.integers(len(pool)))]
        out.append({l: [h]})
    return out
