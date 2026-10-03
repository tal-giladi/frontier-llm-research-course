"""Reference solution for lab 03.1 (Multi-head Latent Attention)."""

from __future__ import annotations

import torch

from frontierlab.attention.ops import attend


def expand_latent(c, w_kv_b, H, d_n, d_v):
    """Per-head keys (no RoPE part) and values from cached latents.

    c (B, S, d_c); w_kv_b (H*(d_n+d_v), d_c), the weight of kv_b_proj, rows ordered head by head as
    [W_UK,h ; W_UV,h]. Returns k_nope (B, H, S, d_n) and v (B, H, S, d_v).
    """
    B, S, _ = c.shape
    kv = (c @ w_kv_b.T).view(B, S, H, d_n + d_v).transpose(1, 2)
    return kv[..., :d_n], kv[..., d_n:]


def absorb_query(q_nope, w_kv_b, H, d_n, d_v):
    """Fold W_UK into the query: q_lat[b,h,t,:] = W_UK,h^T q_nope[b,h,t,:]. Returns (B, H, T, d_c)."""
    w = w_kv_b.view(H, d_n + d_v, -1)
    return torch.einsum("bhtn,hnc->bhtc", q_nope, w[:, :d_n])


def absorbed_attention(q_nope, q_rope, c, k_rope, w_kv_b, H, d_n, d_v, q_pos, k_pos, scale):
    """MLA attention without ever building per-head keys or values. Returns (B, H, T, d_v).

    q_nope (B, H, T, d_n), q_rope (B, H, T, d_r) after RoPE; c (B, S, d_c); k_rope (B, 1, S, d_r)
    after RoPE. Over the latent this is multi-query attention: one key [c ; k_rope] and one value c,
    shared by all heads. W_UV,h is applied to the attended latent afterwards.
    """
    q_lat = absorb_query(q_nope, w_kv_b, H, d_n, d_v)
    qa = torch.cat((q_lat, q_rope), dim=-1)
    ka = torch.cat((c.unsqueeze(1), k_rope), dim=-1)
    y_lat = attend(qa, ka, c.unsqueeze(1), q_pos, k_pos, scale=scale)
    w_uv = w_kv_b.view(H, d_n + d_v, -1)[:, d_n:]
    return torch.einsum("bhtc,hvc->bhtv", y_lat, w_uv)


def kv_elements_per_token(kind, L, K=None, d=None, d_c=None, d_r=None):
    """Cached elements per token for the whole model (all L layers).

    kind "mha"/"gqa"/"mqa": K key/value heads of width d (MQA: K = 1) -> 2·K·d per layer;
    kind "mla": latent d_c plus the shared RoPE key d_r -> d_c + d_r per layer.
    """
    if kind in ("mha", "gqa", "mqa"):
        return L * 2 * (1 if kind == "mqa" else K) * d
    if kind == "mla":
        return L * (d_c + d_r)
    raise ValueError(kind)
