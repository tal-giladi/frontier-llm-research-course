"""Reference solution for lab 06.7 (non-autoregressive and latent reasoning, extension)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def masked_terms(logits: torch.Tensor, x0: torch.Tensor, masked: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """LLaDA Eq. 3 per position, divided by L: (1/t) * 1[masked] * CE(logits, x0) / L. Shapes (B, L, V), (B, L),
    (B, L) bool, (B,). Returns (B, L)."""
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x0.reshape(-1), reduction="none").view(x0.shape)
    return masked.to(ce.dtype) * ce / t[:, None] / x0.shape[1]


def eq6_estimate(ce: torch.Tensor, masked: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    """LLaDA Eq. 6 for one draw: exactly l[b] of the L positions are masked; (L / l) * sum of CE over the masked
    positions, divided by L (nats per token). ce (B, L), masked (B, L) bool, l (B,)."""
    L = ce.shape[1]
    return (L / l.to(ce.dtype)) * (ce * masked.to(ce.dtype)).sum(1) / L


def commit_count(remaining: int, steps_left: int) -> int:
    """Tokens to unmask at this sampling step so that every mask is gone after ``steps_left`` steps."""
    return math.ceil(remaining / steps_left)


def effective_depth(prelude: int, core: int, coda: int, r: int) -> int:
    """Layers a token passes through in a recurrent-depth model with r iterations of the core."""
    return prelude + core * r + coda
