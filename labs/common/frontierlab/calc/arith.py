"""Parameters, active parameters, FLOPs per token and KV-cache bytes from an :class:`ArchSpec` (lesson 01.2).

Conventions (the same as :mod:`frontierlab.flops` and parent course lesson 07.4):

* one multiply-add = 2 FLOPs; training ≈ 3 × forward;
* forward FLOPs per token ≈ 2 × (active parameters used in matmuls) + attention-score FLOPs;
  the embedding lookup costs no matmul FLOPs, the output head does (2·V·C, also when tied);
* attention-score FLOPs per layer for a query that sees ``k`` keys: QKᵀ costs 2·H·d_qk·k and AV costs
  2·H·d_v·k. In training, the average ``k`` over a causal sequence of length T is T/2 for a full
  layer and about w − w²/(2T) for a sliding window w < T. In decoding at context S it is S (full) or
  min(S, w) (sliding);
* linear-attention layers have no T term; their recurrent state update is counted approximately as
  6·H_v·d_k·d_v per token (read, update, read out), which is small next to the projections.

KV-cache elements per token and layer:

* GQA, full layer: K and V for every KV head, ``kv·(d_qk + d_v)``;
* sliding layer: the same, but only the last ``w`` tokens are kept, so the per-token cost falls as S grows;
* MLA (cached in compressed form): the latent ``c_KV`` plus the shared RoPE key, ``kv_lora_rank + qk_rope_dim``;
* linear layer: a fixed-size state, ``H_v·d_k·d_v`` plus the short-convolution buffer.

Every function returns plain Python numbers so a learner can check them by hand.
"""

from __future__ import annotations

from frontierlab.calc.hfconfig import ArchSpec


def attention_params(s: ArchSpec, kind: str) -> int:
    """Parameters of one attention (or linear-attention) block of the given layer kind."""
    C = s.hidden_size
    if kind == "linear":
        kd, vd = s.lin_k_heads * s.lin_k_dim, s.lin_v_heads * s.lin_v_dim
        return (C * (2 * kd + 2 * vd)          # in_proj_qkvz
                + C * 2 * s.lin_v_heads        # in_proj_ba (beta and decay inputs)
                + (2 * kd + vd) * s.lin_conv   # depthwise short convolution
                + 2 * s.lin_v_heads            # A_log, dt_bias
                + s.lin_v_dim                  # gated output norm
                + vd * C)                      # out_proj
    H, KV = s.num_heads, s.num_kv_heads
    if s.attention == "mla":
        qk = s.qk_nope_dim + s.qk_rope_dim
        if s.q_lora_rank:
            q = C * s.q_lora_rank + s.q_lora_rank + s.q_lora_rank * H * qk
        else:
            q = C * H * qk
        kv = C * (s.kv_lora_rank + s.qk_rope_dim) + s.kv_lora_rank + s.kv_lora_rank * H * (s.qk_nope_dim + s.v_head_dim)
        return q + kv + H * s.v_head_dim * C
    d, dv = s.head_dim, s.v_head_dim
    q_out = H * d * (2 if s.q_gate else 1)
    p = C * q_out + C * KV * d + C * KV * dv + H * dv * C
    if s.qkv_bias:
        p += q_out + KV * d + KV * dv
    if s.o_bias:
        p += C
    if s.qk_norm == "per_head":
        p += 2 * d
    elif s.qk_norm == "full":
        p += H * d + KV * d
    if s.sinks:
        p += H
    return p


def expert_params(s: ArchSpec) -> int:
    """One routed expert: SwiGLU gate, up and down (plus biases where the model has them)."""
    C, I = s.hidden_size, s.expert_intermediate
    return 3 * C * I + ((2 * I + C) if s.expert_bias else 0)


def ffn_params(s: ArchSpec, layer: int) -> dict:
    """Feed-forward parameters of one layer: ``total`` and ``active`` (used per token)."""
    C = s.hidden_size
    if not s.is_moe or layer in s.dense_layers:
        p = 3 * C * s.dense_intermediate
        return {"total": p, "active": p}
    router = s.n_experts * C + (s.n_experts if s.router_bias else 0)
    shared = 3 * C * s.shared_intermediate + (C if s.shared_gate else 0)
    e = expert_params(s)
    return {"total": router + shared + s.n_experts * e, "active": router + shared + s.top_k * e}


def param_counts(s: ArchSpec) -> dict:
    """Total and active parameters, split into embedding, head, attention, FFN and norms.

    ``active`` follows the convention most reports use: everything one token touches, including the
    output head; the input embedding is a lookup and is reported separately (``active_with_embedding``
    adds it back, which is how DeepSeek-V3's "37B activated" is counted).
    """
    C, V = s.hidden_size, s.vocab_size
    emb = V * C
    head = 0 if s.tie_embeddings else V * C
    attn = sum(attention_params(s, k) for k in s.layer_kinds)
    ffn = [ffn_params(s, i) for i in range(s.num_layers)]
    norms = s.num_layers * s.norms_per_layer * C + C
    ffn_total, ffn_active = sum(f["total"] for f in ffn), sum(f["active"] for f in ffn)
    total = emb + head + attn + ffn_total + norms
    active = (head if head else emb) + attn + ffn_active + norms     # a tied head reuses the embedding
    return {"total": total, "embedding": emb, "head": head, "attention": attn, "ffn": ffn_total,
            "ffn_active": ffn_active, "norms": norms, "non_embedding": total - emb,
            "active": active, "active_with_embedding": active + (emb if head else 0)}


def _avg_keys_train(kind: str, T: int, w: int | None) -> float:
    if kind == "full" or not w or T <= w:
        return T / 2
    return w - w * w / (2 * T)


def _score_flops_per_key(s: ArchSpec) -> float:
    """FLOPs per (query, key) pair in one layer: QKᵀ plus AV over all heads."""
    if s.attention == "mla":
        return 2 * s.num_heads * (s.qk_nope_dim + s.qk_rope_dim + s.v_head_dim)
    return 2 * s.num_heads * (s.head_dim + s.v_head_dim)


def _linear_state_flops(s: ArchSpec) -> float:
    return 6 * s.lin_v_heads * s.lin_k_dim * s.lin_v_dim


def matmul_params(s: ArchSpec) -> int:
    """Active parameters that take part in a matrix multiply per token (no embedding lookup, no norms)."""
    pc = param_counts(s)
    head = pc["head"] or pc["embedding"]                  # the head is a matmul even when tied
    return pc["attention"] + pc["ffn_active"] + head


def flops_per_token(s: ArchSpec, T: int, training: bool = True) -> float:
    """Average FLOPs per token over a training sequence of length ``T`` (forward, or 3× for training)."""
    fwd = 2 * matmul_params(s)
    for kind in s.layer_kinds:
        if kind == "linear":
            fwd += _linear_state_flops(s)
        else:
            fwd += _score_flops_per_key(s) * _avg_keys_train(kind, T, s.sliding_window)
    return 3 * fwd if training else fwd


def decode_flops_per_token(s: ArchSpec, S: int) -> float:
    """Forward FLOPs to generate one token with ``S`` tokens already in context."""
    fwd = 2 * matmul_params(s)
    for kind in s.layer_kinds:
        if kind == "linear":
            fwd += _linear_state_flops(s)
        else:
            k = S if kind == "full" or not s.sliding_window else min(S, s.sliding_window)
            fwd += _score_flops_per_key(s) * k
    return fwd


def kv_cache_elements(s: ArchSpec, S: int) -> dict:
    """Cached elements for one sequence of ``S`` tokens, per layer kind and in total."""
    out = {"full": 0, "sliding": 0, "linear": 0}
    for kind in s.layer_kinds:
        if kind == "linear":
            kd, vd = s.lin_k_heads * s.lin_k_dim, s.lin_v_heads * s.lin_v_dim
            out["linear"] += s.lin_v_heads * s.lin_k_dim * s.lin_v_dim + (2 * kd + vd) * (s.lin_conv - 1)
            continue
        if s.attention == "mla":
            per_tok = s.kv_lora_rank + s.qk_rope_dim
        else:
            per_tok = s.num_kv_heads * (s.head_dim + s.v_head_dim)
        n = S if kind == "full" or not s.sliding_window else min(S, s.sliding_window)
        out[kind] += n * per_tok
    out["total"] = out["full"] + out["sliding"] + out["linear"]
    return out


def kv_bytes(s: ArchSpec, S: int, bytes_per_elem: float = 2) -> float:
    """Decode-cache bytes for one sequence of ``S`` tokens (BF16 = 2 bytes, FP8 = 1)."""
    return kv_cache_elements(s, S)["total"] * bytes_per_elem


def kv_bytes_per_token(s: ArchSpec, S: int, bytes_per_elem: float = 2) -> float:
    """Average cache bytes per token at context ``S`` (falls with S when some layers are windowed)."""
    return kv_bytes(s, S, bytes_per_elem) / S


def summary_row(s: ArchSpec, T: int = 4096, S: int = 32768) -> dict:
    pc = param_counts(s)
    kinds = {k: s.layer_kinds.count(k) for k in ("full", "sliding", "linear") if k in s.layer_kinds}
    return {"model": s.name, "attention": s.attention, "layers": s.num_layers, "kinds": kinds,
            "heads": s.num_heads, "kv_heads": s.num_kv_heads, "total_B": pc["total"] / 1e9,
            "active_B": pc["active"] / 1e9, "train_GFLOPs_per_tok": flops_per_token(s, T) / 1e9,
            f"kv_KiB_per_tok@{S}": kv_bytes_per_token(s, S) / 1024}
