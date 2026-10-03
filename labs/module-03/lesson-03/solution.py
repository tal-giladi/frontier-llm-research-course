"""Reference solution for lab 03.3 (logit control, sinks and gating)."""

from __future__ import annotations

import torch


def sink_softmax(logits: torch.Tensor, sink: torch.Tensor) -> torch.Tensor:
    """Softmax over keys with one extra per-head logit that takes probability and returns no value.

    logits (B, H, T, S) with -inf where masked; sink (H,). Returns (B, H, T, S); rows sum to 1 - p_sink.
    Subtract the row maximum (including the sink) before exponentiating, so large logits cannot overflow.
    """
    B, H, T, S = logits.shape
    s = sink.view(1, H, 1, 1).expand(B, H, T, 1).to(logits.dtype)
    z = torch.cat((logits, s), dim=-1)
    z = z - z.amax(dim=-1, keepdim=True)
    e = z.exp()
    return (e / e.sum(dim=-1, keepdim=True))[..., :S]


def first_token_mass(probs: torch.Tensor, skip: int = 8) -> float:
    """Mean probability on key position 0, over batch, heads and queries t >= skip. probs (B, H, T, T)."""
    return probs[:, :, skip:, 0].float().mean().item()


def max_logit(logits: torch.Tensor) -> float:
    """Largest finite pre-softmax score; masked entries are -inf and must be ignored."""
    return logits[torch.isfinite(logits)].max().item()


def massive_ratio(h: torch.Tensor) -> float:
    """max |h| divided by median |h| over all entries of one layer's output (B, T, C)."""
    a = h.float().abs().flatten()
    return (a.max() / a.median()).item()
