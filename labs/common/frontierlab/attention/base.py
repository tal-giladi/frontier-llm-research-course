"""Shared attention pieces: RoPE, the decode cache, the causal mask and the attention registry.

Every attention kind is an ``nn.Module`` with the same call signature::

    y = attn(x, positions, cache)      x (B, T, C), positions (T,) long, cache LayerCache | None

``positions`` are absolute token positions (``start .. start+T-1``), so the same module serves a
full-sequence forward (``start = 0``, no cache) and incremental decoding (``start`` = tokens already
in the cache). The module decides what it stores in its ``LayerCache`` (K/V for GQA, a latent for
MLA, a window for sliding attention, a recurrent state for linear attention). The correctness suite
(:mod:`frontierlab.testing`) checks that both paths give the same logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn

ATTENTION: dict[str, type[nn.Module]] = {}


def register(name: str):
    """Class decorator: ``@register("mla")`` makes ``ModelConfig(attention="mla")`` build that class."""
    def deco(cls):
        ATTENTION[name] = cls
        return cls
    return deco


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """RoPE in the "rotate half" convention used by Llama/Qwen (parent course lesson 12.1).

    ``forward(positions)`` returns ``cos, sin`` of shape ``(T, rot_dim)`` for those absolute
    positions; ``apply_rope`` rotates the first ``rot_dim`` channels of each head (all of them
    unless partial RoPE is used).
    """

    def __init__(self, rot_dim: int, theta: float):
        super().__init__()
        self.rot_dim = rot_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, rot_dim, 2, dtype=torch.float64) / rot_dim))
        self.register_buffer("inv_freq", inv_freq.float(), persistent=False)

    def forward(self, positions: torch.Tensor):
        freqs = torch.outer(positions.float(), self.inv_freq.to(positions.device))   # (T, rot/2)
        emb = torch.cat((freqs, freqs), dim=-1)                                      # (T, rot)
        return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape (B, heads, T, hd); only the first ``cos.shape[-1]`` channels rotate."""
    r = cos.shape[-1]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    if r == x.shape[-1]:
        return x * cos + rotate_half(x) * sin
    xr, xp = x[..., :r], x[..., r:]
    return torch.cat((xr * cos + rotate_half(xr) * sin, xp), dim=-1)


def causal_mask(q_pos: torch.Tensor, k_pos: torch.Tensor) -> torch.Tensor:
    """Boolean (Tq, Tk) mask, True where query i may attend key j (``k_pos[j] <= q_pos[i]``)."""
    return k_pos[None, :] <= q_pos[:, None]


class LayerCache(dict):
    """Per-layer decode state. A plain dict so each attention kind stores what it needs."""


class Cache:
    """Decode cache for the whole model: one :class:`LayerCache` per layer plus the next position."""

    def __init__(self, num_layers: int):
        self.layers = [LayerCache() for _ in range(num_layers)]
        self.length = 0          # tokens already processed

    def nbytes(self) -> int:
        """Bytes held by tensors in the cache (the KV-memory number Module 3 compares)."""
        return sum(v.numel() * v.element_size() for layer in self.layers for v in layer.values()
                   if isinstance(v, torch.Tensor))
