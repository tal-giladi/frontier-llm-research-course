"""The project's claim harness: the three decisions every supported causal claim depends on, written so they can
be tested. ``buggy_harness.py`` is a colleague's version of this file with the same API.

* ``effect`` — the normalised effect of an intervention (lesson 17.2): divide by the gap between the *mean*
  clean and the *mean* corrupt metric.
* ``heldout`` — the evaluation prompts: a distribution the candidate was *not* chosen on (other templates and
  other names for IOI; other gaps and query positions for induction).
* ``random_sets`` — the control: random head sets of the *same size* as the candidate, drawn from the heads
  *outside* it, with the *same number of heads in each layer* (``layer_matched=True``, the project's pre-stated
  control). Unmatched sets can pick an early head whose ablation breaks the whole model, which tests "is the
  candidate more important than a random head anywhere?", a weaker and noisier question.
"""

from __future__ import annotations

import numpy as np
import torch

from frontierlab.interp import tasks as T


def effect(m_patched: torch.Tensor, m_clean: torch.Tensor, m_corrupt: torch.Tensor, mode: str = "noise") -> torch.Tensor:
    gap = m_clean.mean() - m_corrupt.mean()
    return (m_clean - m_patched) / gap if mode == "noise" else (m_patched - m_corrupt) / gap


def heldout(kind: str, n: int, seed: int = 0, tok=None):
    """Held-out pairs. ``kind="induction"``: an unseen gap (12) and another query position (2);
    ``kind="ioi"``: the held-out template family with names disjoint from the selection names."""
    if kind == "induction":
        return T.induction_pairs(n, kind="key", gap=12, max_gap=12, query=2, seed=seed + 101)
    return T.as_pairs(T.ioi_prompts(tok, "heldout", n, seed + 1, T.NAMES[12:]))


def selection(kind: str, n: int, seed: int = 0, tok=None):
    if kind == "induction":
        return T.induction_pairs(n, kind="key", seed=seed)
    return T.as_pairs(T.ioi_prompts(tok, "train", n, seed, T.NAMES[:12]))


def random_sets(candidate: dict[int, list[int]], n_layers: int, n_heads: int, n: int, seed: int = 0,
                layer_matched: bool = True) -> list[dict]:
    taken = {(l, h) for l, hs in candidate.items() for h in hs}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        d: dict[int, list[int]] = {}
        if layer_matched:
            for l, hs in candidate.items():
                pool = [h for h in range(n_heads) if (l, h) not in taken]
                d[l] = sorted(int(h) for h in rng.choice(pool, size=len(hs), replace=False))
        else:
            pool = [(l, h) for l in range(n_layers) for h in range(n_heads) if (l, h) not in taken]
            for i in rng.choice(len(pool), size=len(taken), replace=False):
                l, h = pool[i]
                d.setdefault(l, []).append(h)
        out.append(d)
    return out
