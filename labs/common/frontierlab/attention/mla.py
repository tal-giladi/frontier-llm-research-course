"""Multi-head Latent Attention (MLA), DeepSeek-V2 section 2.1 (lesson 03.1).

Keys and values are not cached per head. Each token is compressed into one latent vector ``c`` of
width ``d_c`` (``kv_lora_rank``); per-head keys and values are up-projections of that latent. RoPE
cannot act on those up-projected keys without breaking weight absorption (see below), so each token
also gets one small *decoupled* RoPE key ``k_rope`` of width ``d_r`` (``qk_rope_head_dim``), shared by
all heads. The cache holds only ``c`` and ``k_rope``: ``d_c + d_r`` elements per token and layer.

Equations (DeepSeek-V2 eqs. 9-19, with h the layer input of token t, i a head):

    c_t      = RMSNorm(W_DKV h_t)                       (B, S, d_c)        cached
    k_rope_t = RoPE(W_KR h_t)                            (B, 1, S, d_r)     cached, shared by heads
    k_t,i    = [W_UK,i c_t ; k_rope_t]                   per-head key of width d_n + d_r
    v_t,i    = W_UV,i c_t                                per-head value of width d_v
    q_t,i    = [W_UQ,i h_t ; RoPE(W_QR,i h_t)]           (or through a query latent if q_lora_rank)
    score    = q_t,i . k_s,i / sqrt(d_n + d_r)

Weight absorption: q_nope . (W_UK c) = (W_UK^T q_nope) . c, so the per-head key never has to be built:
project the query into latent space once per token and attend directly over the cached latents. Over
the latent, attention is multi-query attention (one shared "head" of width d_c + d_r) and the value is
the latent itself; W_UV is applied to the attended latent afterwards. If RoPE were applied to
W_UK c, a position-dependent rotation R_s would sit between W_UK and c_s, different for every cached
position s, and W_UK could no longer be folded into the query ("matrix multiplication does not obey a
commutative law", DeepSeek-V2 section 2.1.3). That is why the RoPE part is a separate, uncompressed key.

Names follow Hugging Face ``DeepseekV3Attention`` (``q_proj`` or ``q_a_proj``/``q_a_layernorm``/
``q_b_proj``, ``kv_a_proj_with_mqa``, ``kv_a_layernorm``, ``kv_b_proj``, ``o_proj``). RoPE here uses the
course's rotate-half layout, so weights are not interchangeable with DeepSeek checkpoints as is.

Settings, read from ``cfg.extra`` (defaults in brackets, d = cfg.head_dim):

    kv_lora_rank       d_c  [4 d]      DeepSeek-V2 uses d_c = 4 d_h (Table 1 caption)
    qk_rope_head_dim   d_r  [d / 2]    DeepSeek-V2 uses d_h^R = d_h / 2
    qk_nope_head_dim   d_n  [d]
    v_head_dim         d_v  [d]
    q_lora_rank        r_q  [None]     None = full-rank query projection
    mla_mode                ["naive"]  "naive" builds per-head K/V from the cache; "absorbed" attends over
                                       the latent. Both give the same function (tests/test_attention_m03).

``num_key_value_heads`` is ignored: MLA has no KV heads. QK-norm is not used (DeepSeek-V2 normalises the
latents instead: "additional RMS Norm layers after the compressed latent vectors", in the report's training-stability details).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from frontierlab.attention.base import LayerCache, RotaryEmbedding, apply_rope, register
from frontierlab.attention.ops import attend
from frontierlab.layers.rmsnorm import RMSNorm


def mla_dims(cfg) -> dict:
    """Resolved MLA widths for a config (defaults as in the module docstring)."""
    ex, d = cfg.extra, cfg.head_dim
    return {"d_c": int(ex.get("kv_lora_rank", 4 * d)), "d_r": int(ex.get("qk_rope_head_dim", d // 2)),
            "d_n": int(ex.get("qk_nope_head_dim", d)), "d_v": int(ex.get("v_head_dim", d)),
            "r_q": ex.get("q_lora_rank")}


@register("mla")
class MLAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C, H = cfg.hidden_size, cfg.num_attention_heads
        dims = mla_dims(cfg)
        self.H, self.d_c, self.d_r, self.d_n, self.d_v, self.r_q = (H, dims["d_c"], dims["d_r"], dims["d_n"],
                                                                     dims["d_v"], dims["r_q"])
        if self.d_r % 2:
            raise ValueError("qk_rope_head_dim must be even (RoPE rotates pairs)")
        self.mode = cfg.extra.get("mla_mode", "naive")
        dq = self.d_n + self.d_r
        if self.r_q:
            self.q_a_proj = nn.Linear(C, self.r_q, bias=False)
            self.q_a_layernorm = RMSNorm(self.r_q, eps=cfg.rms_norm_eps)
            self.q_b_proj = nn.Linear(self.r_q, H * dq, bias=False)
        else:
            self.q_proj = nn.Linear(C, H * dq, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(C, self.d_c + self.d_r, bias=False)
        self.kv_a_layernorm = RMSNorm(self.d_c, eps=cfg.rms_norm_eps)
        self.kv_b_proj = nn.Linear(self.d_c, H * (self.d_n + self.d_v), bias=False)
        self.o_proj = nn.Linear(H * self.d_v, C, bias=False)
        self.rope = RotaryEmbedding(self.d_r, cfg.rope_theta)
        self.scale = dq ** -0.5

    # --- pieces ------------------------------------------------------------------------------
    def queries(self, x, cos, sin):
        """q_nope (B, H, T, d_n) and q_rope (B, H, T, d_r) after RoPE."""
        B, T, _ = x.shape
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x))) if self.r_q else self.q_proj(x)
        q = q.view(B, T, self.H, self.d_n + self.d_r).transpose(1, 2)
        q_nope, q_rope = q.split((self.d_n, self.d_r), dim=-1)
        return q_nope, apply_rope(q_rope, cos, sin)

    def latents(self, x, cos, sin):
        """What the cache stores: c (B, T, d_c) after its RMSNorm, k_rope (B, 1, T, d_r) after RoPE."""
        c, k_rope = self.kv_a_proj_with_mqa(x).split((self.d_c, self.d_r), dim=-1)
        return self.kv_a_layernorm(c), apply_rope(k_rope.unsqueeze(1), cos, sin)

    def up_weights(self):
        """W_UK (H, d_n, d_c) and W_UV (H, d_v, d_c), views of kv_b_proj.weight."""
        w = self.kv_b_proj.weight.view(self.H, self.d_n + self.d_v, self.d_c)
        return w[:, :self.d_n], w[:, self.d_n:]

    def naive(self, q_nope, q_rope, c, k_rope, q_pos, k_pos, return_probs=False):
        """Build every head's K and V from the cached latent, then ordinary attention. -> (B, H, T, d_v)."""
        B, S, _ = c.shape
        kv = self.kv_b_proj(c).view(B, S, self.H, self.d_n + self.d_v).transpose(1, 2)
        k_nope, v = kv.split((self.d_n, self.d_v), dim=-1)
        k = torch.cat((k_nope, k_rope.expand(B, self.H, S, self.d_r)), dim=-1)
        q = torch.cat((q_nope, q_rope), dim=-1)
        return attend(q, k, v, q_pos, k_pos, scale=self.scale, return_probs=return_probs)

    def absorbed(self, q_nope, q_rope, c, k_rope, q_pos, k_pos, return_probs=False):
        """Fold W_UK into the query and attend over the latent (multi-query attention); then apply W_UV."""
        w_uk, w_uv = self.up_weights()
        B, H, T, _ = q_nope.shape
        q_lat = torch.einsum("bhtn,hnc->bhtc", q_nope, w_uk)                 # (B, H, T, d_c)
        qa = torch.cat((q_lat, q_rope), dim=-1)                              # (B, H, T, d_c + d_r)
        ka = torch.cat((c.unsqueeze(1), k_rope), dim=-1)                     # (B, 1, S, d_c + d_r)
        # Every head shares the same key and value, so fold the heads into the query rows: one
        # "head" with H*T queries reads the latent cache once (no per-head copy of it).
        rows = qa.reshape(B, 1, H * T, -1)
        out = attend(rows, ka, c.unsqueeze(1), q_pos.repeat(H), k_pos, scale=self.scale,
                     return_probs=return_probs)
        y_lat, p = out if return_probs else (out, None)
        y_lat = y_lat.view(B, H, T, self.d_c)
        y = torch.einsum("bhtc,hvc->bhtv", y_lat, w_uv)                      # (B, H, T, d_v)
        return (y, p.view(B, H, T, -1)) if return_probs else y

    # --- the attention interface ---------------------------------------------------------------
    def forward(self, x: torch.Tensor, positions: torch.Tensor, cache: LayerCache | None = None):
        B, T, _ = x.shape
        cos, sin = self.rope(positions)
        q_nope, q_rope = self.queries(x, cos, sin)
        c, k_rope = self.latents(x, cos, sin)
        k_pos = positions
        if cache is not None:
            if "c_kv" in cache:
                c = torch.cat((cache["c_kv"], c), dim=1)
                k_rope = torch.cat((cache["k_rope"], k_rope), dim=2)
                k_pos = torch.cat((cache["pos"], positions))
            cache["c_kv"], cache["k_rope"], cache["pos"] = c, k_rope, k_pos
        fn = self.absorbed if self.mode == "absorbed" else self.naive
        y = fn(q_nope, q_rope, c, k_rope, positions, k_pos)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.d_v))

    def attention_probs(self, x, positions):
        """Full-sequence attention probabilities (B, H, T, T) and logits, for the probes of lesson 03.3."""
        from frontierlab.attention.ops import attention_logits
        cos, sin = self.rope(positions)
        q_nope, q_rope = self.queries(x, cos, sin)
        c, k_rope = self.latents(x, cos, sin)
        B, S, _ = c.shape
        kv = self.kv_b_proj(c).view(B, S, self.H, self.d_n + self.d_v).transpose(1, 2)
        k = torch.cat((kv[..., :self.d_n], k_rope.expand(B, self.H, S, self.d_r)), dim=-1)
        logits = attention_logits(torch.cat((q_nope, q_rope), -1), k, positions, positions, None, self.scale)
        return torch.softmax(logits.float(), -1), logits


def set_mla_mode(model, mode: str) -> int:
    """Switch every MLA layer of ``model`` to "naive" or "absorbed"; returns how many layers changed."""
    if mode not in ("naive", "absorbed"):
        raise ValueError(mode)
    n = 0
    for m in model.modules():
        if isinstance(m, MLAttention):
            m.mode, n = mode, n + 1
    return n
