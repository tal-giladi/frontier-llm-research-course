"""Lab 03.4 (extension) — head count and head dimension. Fill in the TODOs; run `pytest labs/module-03/lesson-04`."""

from __future__ import annotations

import torch

from frontierlab.attention.base import rotate_half


def attention_flops_per_token(H: int, d_qk: int, d_v: int, keys: float) -> float:
    """Score FLOPs of one layer for one query that sees ``keys`` keys: QK^T (2·H·d_qk) plus AV (2·H·d_v) per key."""
    raise NotImplementedError("TODO 1: attention-score FLOPs per token")


def decode_attention_share(H: int, d_qk: int, d_v: int, S: int, matmul_params_per_layer: float) -> float:
    """Fraction of one layer's decode FLOPs spent on attention scores at context S (the rest: 2 x params)."""
    raise NotImplementedError("TODO 2: share of decode FLOPs in attention")


def partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate only the first r = cos.shape[-1] channels of x (B, H, T, d), rotate-half convention; pass the
    remaining d - r channels through unchanged."""
    raise NotImplementedError("TODO 3: rotate only the first r channels")
