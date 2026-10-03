"""Lab 03.3 — logit control, sinks and gating. Fill in the TODOs; run `pytest labs/module-03/lesson-03` to check."""

from __future__ import annotations

import torch


def sink_softmax(logits: torch.Tensor, sink: torch.Tensor) -> torch.Tensor:
    """Softmax over keys with one extra per-head logit that takes probability and returns no value.

    logits (B, H, T, S) with -inf where masked; sink (H,). Returns (B, H, T, S); rows sum to 1 - p_sink.
    Subtract the row maximum (including the sink) before exponentiating, so large logits cannot overflow.
    """
    raise NotImplementedError("TODO 1: softmax with a learned sink logit per head")


def first_token_mass(probs: torch.Tensor, skip: int = 8) -> float:
    """Mean probability on key position 0, over batch, heads and queries t >= skip. probs (B, H, T, T)."""
    raise NotImplementedError("TODO 2: attention mass on the first key")


def max_logit(logits: torch.Tensor) -> float:
    """Largest finite pre-softmax score; masked entries are -inf and must be ignored."""
    raise NotImplementedError("TODO 3: largest finite logit")


def massive_ratio(h: torch.Tensor) -> float:
    """max |h| divided by median |h| over all entries of one layer's output (B, T, C)."""
    raise NotImplementedError("TODO 4: max |h| over median |h|")
