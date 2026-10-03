"""Grouped-query attention with RoPE and optional QK-norm: the attention of Baseline-0.

Layout and names follow Hugging Face ``Qwen3Attention`` (q_proj, k_proj, v_proj, o_proj, q_norm,
k_norm). Shapes (B batch, T new tokens, S = cached + new tokens):

    q            (B, H,  T, hd)
    k, v         (B, KV, S, hd)      cached in LayerCache["k"], ["v"] after RoPE
    output       (B, T, C)

With no cache and ``positions = 0..T-1`` it is ordinary causal attention; with a cache the new
queries attend to every cached key plus the new keys up to their own position.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from frontierlab.attention.base import LayerCache, RotaryEmbedding, apply_rope, causal_mask, register
from frontierlab.layers.rmsnorm import RMSNorm


@register("gqa")
class GQAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.H, self.KV, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        if self.H % self.KV:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        C = cfg.hidden_size
        self.q_proj = nn.Linear(C, self.H * self.hd, bias=False)
        self.k_proj = nn.Linear(C, self.KV * self.hd, bias=False)
        self.v_proj = nn.Linear(C, self.KV * self.hd, bias=False)
        self.o_proj = nn.Linear(self.H * self.hd, C, bias=False)
        self.q_norm = RMSNorm(self.hd, eps=cfg.rms_norm_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.hd, eps=cfg.rms_norm_eps) if cfg.qk_norm else nn.Identity()
        self.rope = RotaryEmbedding(self.hd, cfg.rope_theta)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, cache: LayerCache | None = None):
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.H, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.KV, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.KV, self.hd).transpose(1, 2)
        cos, sin = self.rope(positions)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k_pos = positions
        if cache is not None:
            if "k" in cache:
                k = torch.cat((cache["k"], k), dim=2)
                v = torch.cat((cache["v"], v), dim=2)
                k_pos = torch.cat((cache["pos"], positions))
            cache["k"], cache["v"], cache["pos"] = k, v, k_pos
        if k.shape[2] == T and torch.equal(k_pos, positions):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.H != self.KV)
        else:
            mask = causal_mask(positions, k_pos)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=self.H != self.KV)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.hd))
