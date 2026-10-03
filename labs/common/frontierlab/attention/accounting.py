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

GQA_FAMILY = ("gqa", "gqa_partial", "sliding", "local_global", "sink", "gated", "gqa-rope-scaled", "gqa-irope")


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

