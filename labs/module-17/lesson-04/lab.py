"""Lab 17.4 — steering and persona vectors (harmless traits only). Fill in the TODOs; run
`pytest labs/module-17/lesson-04`.

``steer_lab.py`` checks your functions against the course's (``frontierlab.interp.steering``) and then extracts,
applies, controls and evaluates a sycophancy vector in Qwen3-0.6B.
"""

from __future__ import annotations

import torch


def mean_diff(pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
    """Difference of means of two sets of activations (N, C) and (M, C) -> (C,), in float32."""
    raise NotImplementedError("TODO 1: difference of means")


def steer_edit(vec: torch.Tensor, alpha: float, start: int = 0):
    """An edit for a ``resid_post.L`` site (B, T, C): return a copy with ``alpha · vec`` added at positions
    ``start`` and later; earlier positions unchanged. Must not modify its input in place."""
    raise NotImplementedError("TODO 2: the steering edit")


def token_kl(logp_ref: torch.Tensor, logp_new: torch.Tensor) -> torch.Tensor:
    """KL(ref ‖ new) per position from log-probabilities (..., V): Σ_v p_ref (log p_ref − log p_new)."""
    raise NotImplementedError("TODO 3: per-token KL")


def flip_rate(base_logodds, steered_logodds) -> float:
    """Fraction of items whose preferred answer (sign of the log-odds) changes under steering."""
    raise NotImplementedError("TODO 4: flip rate")
