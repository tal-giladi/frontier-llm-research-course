"""Shared helpers for the Module 3 labs and project: the arms, and training a Baseline-0 branch.

Importing this file registers every Module 3 attention kind (mla, sliding, local_global, sink, gated,
gqa_partial). Lab scripts add ``labs/module-03`` to ``sys.path`` and ``import m03``.

Arms are defined *relative to the preset* so the same name means the same design at every size:

    b0            the preset unchanged (Baseline-0's GQA, H query heads, K key/value heads, QK-norm)
    gqa-kv        GQA with K/2 key/value heads (KV cache halved; the "cheaper GQA" arm)
    mla           MLA with latent width d_c = K·d and decoupled RoPE key d_r = d/2 (cache about half of b0's)
    local-global  local/global interleaving: Gemma 3's 5 local : 1 global where the depth allows
                  (global_every = 6 for 6+ layers, else 4), window = T/4 of the training sequence
    sliding       every layer local, window = T/4
    sink          one learned sink logit per head (gpt-oss form)
    gated         head-specific elementwise sigmoid gate after attention (Qwen gated attention, G1)
    no-qknorm     Baseline-0 without QK-norm
    wide-heads    H/2 heads of width 2d (same attention width H·d)
    partial-rope  wide-heads with RoPE on the first 25% of each head's channels (Qwen3-Next style)

``match_params=True`` changes the SwiGLU width so non-embedding parameters equal the preset's
(equal-parameters axis); :func:`equal_flops_steps` gives the step count for the equal-FLOPs axis.
"""

from __future__ import annotations

from frontierlab.attention import accounting, gated, headshape, mla, sliding  # noqa: F401  (registers kinds)
from frontierlab.model import PRESETS, ModelConfig

ARMS = ("b0", "gqa-kv", "mla", "local-global", "sliding", "sink", "gated", "no-qknorm", "wide-heads", "partial-rope")


def arm_config(arm: str, base: ModelConfig, T: int, match_params: bool = False) -> ModelConfig:
    """The config of ``arm`` built from a preset config ``base`` trained at sequence length ``T``."""
    H, K, d, L = base.num_attention_heads, base.num_key_value_heads, base.head_dim, base.num_hidden_layers
    if arm == "b0":
        cfg = base
    elif arm == "gqa-kv":
        cfg = base.with_(num_key_value_heads=max(1, K // 2))
    elif arm == "mla":
        cfg = base.with_(attention="mla", extra={"kv_lora_rank": K * d, "qk_rope_head_dim": d // 2})
    elif arm == "local-global":
        cfg = base.with_(attention="local_global",
                         extra={"global_every": 6 if L >= 6 else 4, "window": max(16, T // 4)})
    elif arm == "sliding":
        cfg = base.with_(attention="sliding", extra={"window": max(16, T // 4)})
    elif arm == "sink":
        cfg = base.with_(attention="sink")
    elif arm == "gated":
        cfg = base.with_(attention="gated", extra={"gate": "elementwise"})
    elif arm == "no-qknorm":
        cfg = base.with_(qk_norm=False)
    elif arm == "wide-heads":
        cfg = base.with_(num_attention_heads=H // 2, num_key_value_heads=max(1, K // 2), head_dim=2 * d)
    elif arm == "partial-rope":
        cfg = base.with_(attention="gqa_partial", num_attention_heads=H // 2, num_key_value_heads=max(1, K // 2),
                         head_dim=2 * d, extra={"rope_fraction": 0.25})
    else:
        raise KeyError(f"unknown arm {arm!r}; known: {ARMS}")
    if match_params and arm != "b0":
        cfg, _ = accounting.match_intermediate(cfg, accounting.param_counts(base)["non_embedding"])
    return cfg


def preset_config(preset: str, vocab_size: int) -> ModelConfig:
    return PRESETS[preset](vocab_size=vocab_size)


def equal_flops_steps(arm_cfg: ModelConfig, base: ModelConfig, base_steps: int, T: int) -> int:
    return accounting.equal_flops_steps(base, arm_cfg, base_steps, T)


def describe(cfg: ModelConfig, T: int, S: int = 32768) -> str:
    pc = accounting.param_counts(cfg)
    return (f"{cfg.attention:<13s} H={cfg.num_attention_heads} K={cfg.num_key_value_heads} d={cfg.head_dim} "
            f"I={cfg.intermediate_size} extra={cfg.extra}  non-emb {pc['non_embedding'] / 1e6:.3f}M  "
            f"train {accounting.flops_per_token(cfg, T) / 1e6:.2f} MFLOP/token at T={T}  "
            f"KV {accounting.kv_bytes(cfg, S) / S:,.0f} B/token at S={S} (BF16)")


def load_run(run, device="cpu"):
    """A trained model from ``run/checkpoint.pt`` (any Module 3 kind) and its run card."""
    from pathlib import Path

    import torch
    import yaml

    from frontierlab.model import LM
    run = Path(run)
    ck = torch.load(run / "checkpoint.pt", map_location=device, weights_only=False)
    model = LM(ModelConfig(**ck["config"])).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), yaml.safe_load((run / "run_card.yaml").read_text())
