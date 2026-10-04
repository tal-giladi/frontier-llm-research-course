"""Reference solution for lab 06.5 (elastic architectures, extension)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nested_ffn(x, gate_w, up_w, down_w, m: int):
    """MatFormer granularity with the first m neurons: W_down[:, :m] (silu(W_gate[:m] x) * (W_up[:m] x))."""
    return F.linear(F.silu(F.linear(x, gate_w[:m])) * F.linear(x, up_w[:m]), down_w[:, :m])


def mix_n_match(widths: list[int], num_layers: int, budget: float) -> list[int]:
    """Per-layer widths with mean closest to ``budget``: the first layers at granularity i, the rest at i + 1."""
    best, err = None, float("inf")
    for lo in range(len(widths)):
        hi = min(len(widths) - 1, lo + 1)
        for split in range(num_layers + 1):
            cfg = [widths[lo]] * split + [widths[hi]] * (num_layers - split)
            e = abs(sum(cfg) / num_layers - budget)
            if e < err:
                best, err = cfg, e
    return best


def consistency(a: torch.Tensor, b: torch.Tensor) -> float:
    """Share of positions where two models' greedy next-token predictions agree. a, b: (B, T) token ids."""
    return float((a == b).float().mean())


def ple_params(vocab_per_layer: int, num_layers: int, dim: int) -> int:
    """Parameters of the per-layer embedding table: one row of num_layers x dim per token id."""
    return vocab_per_layer * num_layers * dim
