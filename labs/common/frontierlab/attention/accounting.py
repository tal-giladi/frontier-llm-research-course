"""Parameters, FLOPs and KV-cache bytes for every attention kind of the course model (Module 3).

``frontierlab.flops`` and ``frontierlab.model.param_counts`` know only Baseline-0's GQA. This module
handles any ``ModelConfig`` whose attention kind is registered, so comparisons at equal parameters and
equal training FLOPs (lesson 01.3) can be set up for MLA, local/global, gated and sink variants.

* Parameters are counted from the real attention modules (one layer at a time on the CPU) plus the
  known SwiGLU, norm and embedding shapes, so they are exact by construction (a test checks the total).
* FLOPs follow the course convention (parent lesson 07.4; ``frontierlab.flops``): forward per token
  = 2 × non-embedding parameters + the output head (2·V·C when tied) + attention-score FLOPs; training
  = 3 × forward. For a Baseline-0 config the result equals ``frontierlab.flops.flops_per_token`` exactly.
* Attention-score FLOPs per (query, key) pair and layer:

      GQA family      4·H·d                         (QKᵀ 2·H·d, AV 2·H·d)
      MLA, naive      2·H·(d_n + d_r) + 2·H·d_v
      MLA, absorbed   2·H·(d_c + d_r) + 2·H·d_c     (scores and values over the latent)

  with the average number of keys per query T/2 for a full layer and w − w²/(2T) for a window w < T
  in training, S or min(S, w) at decode (lesson 01.2).
* MLA decode has one more term. The naive path re-expands every cached latent with ``kv_b_proj`` at
  every step: 2·S·d_c·H·(d_n + d_v) FLOPs per new token. The absorbed path never does.
* KV-cache elements per token and layer: 2·KV·d for the GQA family (a local layer keeps at most w
  tokens), d_c + d_r for MLA. Our implementation also stores the int64 position of every cached token
  in every layer (8 bytes); :func:`kv_bytes` excludes that bookkeeping, :func:`cache_bytes` includes it,
  and ``Cache.nbytes()`` measures the latter.
"""

from __future__ import annotations

import torch

from frontierlab.model.config import ModelConfig

GQA_FAMILY = ("gqa", "gqa_partial", "sliding", "local_global", "sink", "gated", "gqa-rope-scaled", "gqa-irope", "gqa-softcap")


def _import_kinds():
    """Register every Module 3 kind (until the main session adds them to attention/__init__.py)."""
    from frontierlab.attention import gated, headshape, mla, sliding  # noqa: F401


def attention_module(cfg: ModelConfig, layer_idx: int = 0):
    """One attention block of the given layer, built on the CPU (small: only this layer's weights)."""
    _import_kinds()
    from frontierlab.attention.base import ATTENTION
    from frontierlab.model.lm import _takes_layer
    cls = ATTENTION[cfg.attention]
    return cls(cfg, layer_idx=layer_idx) if _takes_layer(cls) else cls(cfg)


def param_counts(cfg: ModelConfig) -> dict:
    """Exact counts for ``LM(cfg)``: each layer's attention block is instantiated and counted; the SwiGLU
    (3·C·I), the two RMSNorm gains per layer, the final norm, the embedding and (if untied) the head follow
    the structure of :mod:`frontierlab.model.lm`. Tests check the total against the real model."""
    C, L, V = cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size
    attn = [sum(p.numel() for p in attention_module(cfg, i).parameters()) for i in range(L)]
    mlp = 3 * C * cfg.intermediate_size
    emb = V * C
    head = 0 if cfg.tie_word_embeddings else V * C
    non_emb = sum(attn) + L * (mlp + 2 * C) + C + head
    return {"total": non_emb + emb, "embedding": emb, "non_embedding": non_emb, "attention_per_layer": attn,
            "attention": sum(attn), "mlp_per_layer": mlp}


def layer_kinds(cfg: ModelConfig) -> list[str]:
    """"local" or "global" per layer, for any kind."""
    if cfg.attention == "local_global":
        from frontierlab.attention.sliding import layer_types
        return layer_types(cfg)
    if cfg.attention == "sliding" or (cfg.attention in ("sink", "gated") and cfg.extra.get("window")):
        return ["local"] * cfg.num_hidden_layers
    return ["global"] * cfg.num_hidden_layers


def window(cfg: ModelConfig) -> int | None:
    if any(k == "local" for k in layer_kinds(cfg)):
        return int(cfg.extra.get("window", 128))
    return None


def _mla(cfg):
    from frontierlab.attention.mla import mla_dims
    return mla_dims(cfg)


def flops_per_key(cfg: ModelConfig, mode: str = "naive") -> float:
    """Attention-score FLOPs for one (query, key) pair in one layer."""
    H = cfg.num_attention_heads
    if cfg.attention == "mla":
        d = _mla(cfg)
        if mode == "absorbed":
            return 2 * H * (d["d_c"] + d["d_r"]) + 2 * H * d["d_c"]
        return 2 * H * (d["d_n"] + d["d_r"]) + 2 * H * d["d_v"]
    if cfg.attention not in GQA_FAMILY:
        raise KeyError(f"no FLOP model for attention {cfg.attention!r}")
    return 4 * H * cfg.head_dim


def avg_keys_train(T: int, w: int | None) -> float:
    """Average keys per query over a causal training sequence of length T (continuous approximation)."""
    if w is None or T <= w:
        return T / 2
    return w - w * w / (2 * T)


def _fwd_matmul(cfg: ModelConfig) -> float:
    head = 2 * cfg.vocab_size * cfg.hidden_size if cfg.tie_word_embeddings else 0   # untied head is in N
    return 2 * param_counts(cfg)["non_embedding"] + head


def flops_per_token(cfg: ModelConfig, T: int, training: bool = True) -> float:
    """Average training (or forward) FLOPs per token at sequence length T. Same convention as frontierlab.flops."""
    if cfg.attention in M05_KINDS:                       # Module 5 kinds: linear, hybrid, DSA
        return m05_flops_per_token(cfg, T, training)
    w = window(cfg)
    per_key = flops_per_key(cfg, "naive")
    attn = sum(per_key * avg_keys_train(T, w if kind == "local" else None) for kind in layer_kinds(cfg))
    fwd = _fwd_matmul(cfg) + attn
    return 3 * fwd if training else fwd


def decode_flops_per_token(cfg: ModelConfig, S: int, mode: str = "naive") -> float:
    """Forward FLOPs to generate one token with S tokens in the cache."""
    w = window(cfg)
    fwd = _fwd_matmul(cfg)
    per_key = flops_per_key(cfg, mode)
    for kind in layer_kinds(cfg):
        keys = min(S, w) if (kind == "local" and w) else S
        fwd += per_key * keys
        if cfg.attention == "mla" and mode == "naive":
            d = _mla(cfg)
            fwd += 2 * keys * d["d_c"] * cfg.num_attention_heads * (d["d_n"] + d["d_v"])
    return fwd


def kv_elements_per_token_layer(cfg: ModelConfig) -> int:
    """Cached elements per token in one (global) layer."""
    if cfg.attention == "mla":
        d = _mla(cfg)
        return d["d_c"] + d["d_r"]
    return 2 * cfg.num_key_value_heads * cfg.head_dim


def kv_bytes(cfg: ModelConfig, S: int, bytes_per: float = 2, batch: int = 1) -> float:
    """K/V (or latent) bytes held after S tokens, for ``batch`` sequences. Excludes position bookkeeping."""
    w, e = window(cfg), kv_elements_per_token_layer(cfg)
    toks = sum(min(S, w) if (kind == "local" and w) else S for kind in layer_kinds(cfg))
    return batch * toks * e * bytes_per


def cache_bytes(cfg: ModelConfig, S: int, bytes_per: float = 2, batch: int = 1) -> float:
    """What ``Cache.nbytes()`` reports for this implementation: kv_bytes plus 8 bytes per cached position per layer."""
    w = window(cfg)
    toks = sum(min(S, w) if (kind == "local" and w) else S for kind in layer_kinds(cfg))
    return kv_bytes(cfg, S, bytes_per, batch) + 8 * toks


def match_intermediate(cfg: ModelConfig, target_non_embedding: int) -> tuple[ModelConfig, float]:
    """Change ``intermediate_size`` so non-embedding parameters match a target (equal-parameters axis).

    Non-embedding parameters are linear in the SwiGLU width I: each unit of I adds 3·C·L parameters.
    Returns the new config and the remaining relative difference (typically < 0.1%).
    """
    ne0 = param_counts(cfg)["non_embedding"]
    per_unit = 3 * cfg.hidden_size * cfg.num_hidden_layers
    I = max(1, cfg.intermediate_size + round((target_non_embedding - ne0) / per_unit))
    new = cfg.with_(intermediate_size=I)
    ne = param_counts(new)["non_embedding"]
    return new, (ne - target_non_embedding) / target_non_embedding


def equal_flops_steps(base: ModelConfig, other: ModelConfig, base_steps: int, T: int) -> int:
    """Steps for ``other`` so its training FLOPs equal ``base`` trained ``base_steps`` steps (same tokens per step)."""
    return max(1, round(base_steps * flops_per_token(base, T) / flops_per_token(other, T)))


def summary(cfg: ModelConfig, T: int, S_list=(32768, 131072, 1048576), bytes_per: float = 2) -> dict:
    pc = param_counts(cfg)
    return {"attention": cfg.attention, "non_embedding": pc["non_embedding"], "total": pc["total"],
            "train_flops_per_token": flops_per_token(cfg, T),
            **{f"kv_bytes_per_token@{S}": kv_bytes(cfg, S, bytes_per) / S for S in S_list}}


def human_bytes(n: float) -> str:
    for unit, k in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if abs(n) >= k:
            return f"{n / k:.2f} {unit}"
    return f"{n:.0f} B"



# =============================================================================================
# Module 5 additions (lessons 05.1-05.4): linear ("gdn", "kda"), "hybrid" and "dsa" kinds.
# New functions only; nothing above is changed. ``param_counts`` above already works for these kinds
# (it instantiates each layer) once their modules are imported, which ``_import_m05`` does.
#
# Mixing FLOPs per token and layer, forward (projections are in 2·N like every other kind):
#
#   full (GQA family)   4·H·d · keys                    keys = T/2 in training, S at decode
#   dsa                 H_I·(2·d_I + 2) · keys          lightning indexer over ALL keys: the O(L^2) term
#                     + 4·H·d · keys_k                  keys_k = k - k^2/(2T) for T > k (else T/2); min(S, k) at decode
#                       (dsa_impl="mask": the main attention costs 4·H·d·keys, as our mask path really does)
#   linear, recurrent   H·7·d_k·d_v                     decay, S^T k, rank-1 update, S^T q
#   linear, chunked     H·(6·d_k·d_v + C·(4.5·d_k + 2·d_v))   approximate count of delta_rule_chunked:
#                       pairwise A and M (3·C·d_k), decay exps (C·d_k/2), two triangular solves
#                       (C·(d_k + d_v)), M·E (C·d_v), three d_k x d_v products (W·S_0, Q~·S_0, K^T E)
#   short convolution   2·K·H·(2·d_k + d_v)
#
# Cache / state bytes per sequence and layer:
#   full  2·KV·d·S·b      dsa  (2·KV·d + d_I)·S·b      linear  H·d_k·d_v·b_state + (K-1)·H·(2·d_k + d_v)·b
# plus 8 bytes per cached position per full/dsa layer for our int64 position bookkeeping.
# =============================================================================================

M05_KINDS = ("gdn", "kda", "hybrid", "dsa")


def _import_m05():
    from frontierlab.attention import deltanet, dsa, hybrid  # noqa: F401  (register gdn, kda, dsa, hybrid)
    _import_kinds()


def m05_layer_kinds(cfg: ModelConfig) -> list[str]:
    """"linear", "full", "dsa" (or "local" for Module 3 windowed kinds) for every layer."""
    _import_m05()
    kind = _m05_family(cfg.attention)
    if kind == "linear":
        return ["linear"] * cfg.num_hidden_layers
    if kind == "dsa":
        return ["dsa"] * cfg.num_hidden_layers
    if cfg.attention == "hybrid":
        from frontierlab.attention.hybrid import hybrid_layer_types
        full = "dsa" if cfg.extra.get("full_kind", "gqa") == "dsa" else "full"
        return [full if t == "full" else "linear" for t in hybrid_layer_types(cfg)]
    return ["full" if k == "global" else k for k in layer_kinds(cfg)]


def _m05_family(name: str) -> str | None:
    """"linear" / "dsa" for the Module 5 kinds and any registered subclass of them (lab variants), else None."""
    from frontierlab.attention.base import ATTENTION
    from frontierlab.attention.deltanet import GatedDeltaAttention
    from frontierlab.attention.dsa import DSAttention
    cls = ATTENTION.get(name)
    if cls is not None and issubclass(cls, GatedDeltaAttention):
        return "linear"
    if cls is not None and issubclass(cls, DSAttention):
        return "dsa"
    return None


def linear_dims(cfg: ModelConfig) -> dict:
    ex = cfg.extra
    dk = int(ex.get("linear_head_dim", cfg.head_dim))
    return {"H": int(ex.get("linear_heads", cfg.num_attention_heads)), "d_k": dk, "d_v": dk,
            "K": int(ex.get("conv_size", 4)), "C": int(ex.get("chunk", 16))}


def dsa_dims(cfg: ModelConfig) -> dict:
    ex = cfg.extra
    return {"H_I": int(ex.get("index_heads", 4)), "d_I": int(ex.get("index_head_dim", 32)),
            "k": int(ex.get("index_topk", 64))}


def _full_cfg(cfg: ModelConfig) -> ModelConfig:
    """The config a full layer of a hybrid is built from."""
    if cfg.attention == "hybrid":
        return cfg.with_(attention=cfg.extra.get("full_kind", "gqa"))
    return cfg


def linear_mix_flops(cfg: ModelConfig, form: str = "chunked") -> float:
    """Mixing FLOPs per token of one linear layer (forward), including the short convolution."""
    d = linear_dims(cfg)
    H, dk, dv, K, C = d["H"], d["d_k"], d["d_v"], d["K"], d["C"]
    core = 7 * dk * dv if form == "recurrent" else 6 * dk * dv + C * (4.5 * dk + 2 * dv)
    conv = 2 * K * H * (2 * dk + dv) if K > 1 else 0
    return H * core + conv


def indexer_flops_per_key(cfg: ModelConfig) -> float:
    """Lightning-indexer FLOPs per (query, key) pair: H_I dot products of width d_I, ReLU, weighted sum."""
    d = dsa_dims(cfg)
    return d["H_I"] * (2 * d["d_I"] + 2)


def _gqa_per_key(cfg: ModelConfig) -> float:
    c = _full_cfg(cfg)
    if c.attention == "hybrid" or _m05_family(c.attention):
        c = c.with_(attention="gqa")
    return flops_per_key(c, "naive")


def m05_mix_flops(cfg: ModelConfig, T: int, *, linear_form: str = "chunked", dsa_impl: str = "ideal") -> dict:
    """Forward mixing FLOPs per token, averaged over a causal sequence of length T, by layer kind."""
    out = {"linear": 0.0, "full": 0.0, "dsa_indexer": 0.0, "dsa_attention": 0.0}
    for kind in m05_layer_kinds(cfg):
        if kind == "linear":
            out["linear"] += linear_mix_flops(cfg, linear_form)
        elif kind == "dsa":
            k = dsa_dims(cfg)["k"]
            out["dsa_indexer"] += indexer_flops_per_key(cfg) * avg_keys_train(T, None)
            keys = avg_keys_train(T, None) if dsa_impl == "mask" else avg_keys_train(T, k)
            out["dsa_attention"] += _gqa_per_key(cfg) * keys
        else:
            w = window(cfg) if kind == "local" else None
            out["full"] += _gqa_per_key(cfg) * avg_keys_train(T, w)
    return out


def m05_flops_per_token(cfg: ModelConfig, T: int, training: bool = True, **kw) -> float:
    """Training (or forward) FLOPs per token for any kind, Module 5 kinds included. Same convention as
    ``flops_per_token``, and equal to it for Baseline-0 and the Module 3 kinds (tested)."""
    _import_m05()
    fwd = _fwd_matmul(cfg) + sum(m05_mix_flops(cfg, T, **kw).values())
    return 3 * fwd if training else fwd


def m05_decode_flops_per_token(cfg: ModelConfig, S: int, *, linear_form: str = "recurrent") -> float:
    """Forward FLOPs to generate one token with S tokens of context."""
    _import_m05()
    fwd = _fwd_matmul(cfg)
    for kind in m05_layer_kinds(cfg):
        if kind == "linear":
            fwd += linear_mix_flops(cfg, linear_form)
        elif kind == "dsa":
            fwd += indexer_flops_per_key(cfg) * S + _gqa_per_key(cfg) * min(S, dsa_dims(cfg)["k"])
        else:
            w = window(cfg) if kind == "local" else None
            fwd += _gqa_per_key(cfg) * (min(S, w) if w else S)
    return fwd


def linear_state_bytes(cfg: ModelConfig, bytes_per: float = 2, state_bytes: float = 4) -> float:
    """Bytes one linear layer keeps per sequence: the d_k x d_v state per head plus the conv tail."""
    d = linear_dims(cfg)
    conv = (d["K"] - 1) * d["H"] * (2 * d["d_k"] + d["d_v"]) * bytes_per if d["K"] > 1 else 0
    return d["H"] * d["d_k"] * d["d_v"] * state_bytes + conv


def m05_cache_bytes(cfg: ModelConfig, S: int, bytes_per: float = 2, state_bytes: float = 4, batch: int = 1,
                    include_pos: bool = False) -> float:
    """Decode-cache bytes after S tokens. ``include_pos=True`` with bytes_per = state_bytes = the model's
    element size reproduces ``Cache.nbytes()`` exactly (tested)."""
    _import_m05()
    full_cfg = _full_cfg(cfg)
    kv_e = 2 * full_cfg.num_key_value_heads * full_cfg.head_dim
    w = window(cfg)
    total = 0.0
    for kind in m05_layer_kinds(cfg):
        if kind == "linear":
            total += linear_state_bytes(cfg, bytes_per, state_bytes)
        elif kind == "dsa":
            total += S * (kv_e + dsa_dims(cfg)["d_I"]) * bytes_per + (8 * S if include_pos else 0)
        else:
            toks = min(S, w) if (kind == "local" and w) else S
            total += toks * kv_e * bytes_per + (8 * toks if include_pos else 0)
    return batch * total


def dsa_crossover_length(cfg: ModelConfig) -> float:
    """Training context L above which a DSA layer does fewer forward mixing FLOPs per token than a dense
    layer of the same shape. FLOPs only: memory traffic, top-k and kernel overheads are not in it.

    Dense: a·L/2 with a = 4·H·d. DSA (L > k): c·L/2 + a·(k - k^2/(2L)) with c = H_I·(2·d_I + 2).
    Equal when (a - c)·L^2/2 - a·k·L + a·k^2/2 = 0; the larger root is returned (inf if c >= a)."""
    import math
    a, c, k = _gqa_per_key(cfg), indexer_flops_per_key(cfg), dsa_dims(cfg)["k"]
    if c >= a:
        return float("inf")
    qa, qb, qc = (a - c) / 2, -a * k, a * k * k / 2
    return (-qb + math.sqrt(qb * qb - 4 * qa * qc)) / (2 * qa)


def linear_crossover_length(cfg: ModelConfig, form: str = "chunked") -> float:
    """Training context L above which one linear layer does fewer mixing FLOPs per token than one full
    GQA layer of the same width: linear_mix_flops = 4·H·d·L/2."""
    return 2 * linear_mix_flops(cfg, form) / _gqa_per_key(cfg)


def compressed_kv_entries(S: int, m: int, window: int = 0) -> int:
    """Lesson 05.4: KV entries a compressed layer holds after S tokens: one per complete block of m
    tokens, plus an uncompressed window of the most recent ``window`` tokens (CSA-style)."""
    return S // m + min(S, window)
