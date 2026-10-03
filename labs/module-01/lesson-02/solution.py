"""Reference solution for lab 01.2."""

from __future__ import annotations

from frontierlab.calc.arith import attention_params as _linear_or_reference
from frontierlab.calc.arith import ffn_params
from frontierlab.calc.hfconfig import ArchSpec


def attention_params(s: ArchSpec) -> int:
    C, H, KV = s.hidden_size, s.num_heads, s.num_kv_heads
    if s.attention == "mla":
        dqk = s.qk_nope_dim + s.qk_rope_dim
        q = (C * s.q_lora_rank + s.q_lora_rank + s.q_lora_rank * H * dqk) if s.q_lora_rank else C * H * dqk
        kv = (C * (s.kv_lora_rank + s.qk_rope_dim) + s.kv_lora_rank
              + s.kv_lora_rank * H * (s.qk_nope_dim + s.v_head_dim))
        return q + kv + H * s.v_head_dim * C
    d, dv = s.head_dim, s.v_head_dim
    q_out = H * d * (2 if s.q_gate else 1)
    p = C * q_out + C * KV * (d + dv) + H * dv * C
    p += (q_out + KV * (d + dv)) if s.qkv_bias else 0
    p += C if s.o_bias else 0
    p += {"none": 0, "per_head": 2 * d, "full": H * d + KV * d}[s.qk_norm]
    return p + (H if s.sinks else 0)


def kv_elements(s: ArchSpec, S: int) -> int:
    per_tok = (s.kv_lora_rank + s.qk_rope_dim) if s.attention == "mla" else s.num_kv_heads * (s.head_dim + s.v_head_dim)
    total = 0
    for kind in s.layer_kinds:
        if kind == "full":
            total += S * per_tok
        elif kind == "sliding":
            total += min(S, s.sliding_window) * per_tok
    return total


def active_params(s: ArchSpec) -> int:
    C = s.hidden_size
    attn = sum(_linear_or_reference(s, "linear") if k == "linear" else attention_params(s) for k in s.layer_kinds)
    ffn = sum(ffn_params(s, i)["active"] for i in range(s.num_layers))
    norms = s.num_layers * s.norms_per_layer * C + C
    return attn + ffn + norms + s.vocab_size * C
