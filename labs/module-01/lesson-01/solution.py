"""Reference solution for lab 01.1."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from frontierlab.attention.base import apply_rope, register
from frontierlab.attention.gqa import GQAttention
from frontierlab.model.config import ModelConfig


def count_params(cfg: ModelConfig) -> int:
    C, L, H, KV, d, I, V = (cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads,
                            cfg.num_key_value_heads, cfg.head_dim, cfg.intermediate_size, cfg.vocab_size)
    attn = 2 * C * H * d + 2 * C * KV * d + (2 * d if cfg.qk_norm else 0)
    per_layer = attn + 3 * C * I + 2 * C
    head = 0 if cfg.tie_word_embeddings else V * C
    return L * per_layer + C + V * C + head


def attend(q, k, v, q_pos, k_pos):
    mask = k_pos[None, :] <= q_pos[:, None]                       # (Tq, Tk), True = may attend
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=q.shape[1] != k.shape[1])


@register("gqa-lab-solution")
class LabGQAttention(GQAttention):
    def forward(self, x, positions, cache=None):
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.H, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.KV, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.KV, self.hd).transpose(1, 2)
        cos, sin = self.rope(positions)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k_pos = positions
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat((cache["k"], k), 2), torch.cat((cache["v"], v), 2)
                k_pos = torch.cat((cache["pos"], positions))
            cache["k"], cache["v"], cache["pos"] = k, v, k_pos
        y = attend(q, k, v, positions, k_pos)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.hd))
