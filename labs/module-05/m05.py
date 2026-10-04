"""Shared helpers for the Module 5 labs and project: the arms, loading runs, and the course's sizes.

Importing this file registers every Module 5 attention kind ("gdn", "kda", "hybrid", "dsa") and the
Module 3 and 4 kinds. Lab scripts add ``labs/module-05`` to ``sys.path`` and ``import m05``.

Arms are defined relative to a preset trained at sequence length T, so the same name means the same
design at every size:

    b0            the preset unchanged (Baseline-0's GQA, every layer full attention)
    hybrid-kda    3 KDA layers : 1 full GQA layer (layer i full when (i + 1) % 4 == 0), Kimi-Linear-style
                  ratio; KDA heads = query heads, head dim = the preset's
    hybrid-gdn    the same layout with Gated DeltaNet (one scalar decay per head; Qwen3-Next-style linear
                  layers). NOTE: it also differs from hybrid-kda in the output-gate activation (SiLU vs sigmoid)
    hybrid-kda-silu  hybrid-kda with the SiLU output gate: against hybrid-gdn only the decay granularity
                  differs; against hybrid-kda only the output gate (lesson 05.1, step 5)
    linear-kda    every layer KDA, no full attention at all (the extreme; MiniMax's worry made concrete)
    dsa           every layer DSA-style selection over Baseline-0's GQA: indexer with 4 heads of width 32,
                  top-k = T / 8 (DeepSeek-V3.2: 2,048 of 128K, i.e. 1/64)

``chunk`` (the chunked-parallel block) is 16 on the CPU path and 64 on the GPU path (flash-linear-attention
uses 64). ``match_params=True`` changes the SwiGLU width so non-embedding parameters equal the preset's.
"""

from __future__ import annotations

from pathlib import Path

import frontierlab.longctx  # noqa: F401  (registers the Module 4 kinds)
from frontierlab.attention import accounting, deltanet, dsa, gated, hybrid, mla, sliding  # noqa: F401
from frontierlab.model import PRESETS, ModelConfig

ARMS = ("b0", "hybrid-kda", "hybrid-gdn", "hybrid-kda-silu", "linear-kda", "dsa")


def arm_config(arm: str, base: ModelConfig, T: int, match_params: bool = False, chunk: int = 16,
               topk: int | None = None, linear_mode: str = "chunked") -> ModelConfig:
    lin = {"chunk": chunk, "linear_mode": linear_mode}
    if arm == "b0":
        cfg = base
    elif arm == "hybrid-kda":
        cfg = base.with_(attention="hybrid", extra={"full_every": 4, "linear_kind": "kda", "full_kind": "gqa", **lin})
    elif arm == "hybrid-gdn":
        cfg = base.with_(attention="hybrid", extra={"full_every": 4, "linear_kind": "gdn", "full_kind": "gqa", **lin})
    elif arm == "hybrid-kda-silu":    # KDA with Gated DeltaNet's SiLU output gate: only the decay granularity differs
        cfg = base.with_(attention="hybrid", extra={"full_every": 4, "linear_kind": "kda", "full_kind": "gqa",
                                                    "out_gate": "silu", **lin})
    elif arm == "linear-kda":
        cfg = base.with_(attention="kda", extra=lin)
    elif arm == "dsa":
        cfg = base.with_(attention="dsa", extra={"index_topk": topk or max(8, T // 8), "index_heads": 4,
                                                 "index_head_dim": 32})
    else:
        raise KeyError(f"unknown arm {arm!r}; known: {ARMS}")
    if match_params and arm not in ("b0", "dsa"):
        cfg, _ = accounting.match_intermediate(cfg, accounting.param_counts(base)["non_embedding"])
    return cfg


def preset_config(preset: str, vocab_size: int) -> ModelConfig:
    return PRESETS[preset](vocab_size=vocab_size)


def describe(cfg: ModelConfig, T: int, S: int = 32768) -> str:
    pc = accounting.param_counts(cfg)
    kinds = accounting.m05_layer_kinds(cfg)
    layout = "".join({"linear": "L", "full": "F", "dsa": "D", "local": "l"}[k] for k in kinds)
    return (f"{cfg.attention:<7s} layers {layout}  I={cfg.intermediate_size}  non-emb {pc['non_embedding'] / 1e6:.3f}M  "
            f"train {accounting.m05_flops_per_token(cfg, T) / 1e6:.2f} MFLOP/token at T={T}  "
            f"cache {accounting.m05_cache_bytes(cfg, S) / 2**20:.2f} MiB at S={S} (BF16 K/V, fp32 state)")


def load_run(run, device="cpu"):
    """A trained model from ``run/checkpoint.pt`` (any registered kind) and its run card (or None)."""
    import torch
    import yaml

    from frontierlab.model import LM
    run = Path(run)
    ck = torch.load(run / "checkpoint.pt" if run.is_dir() else run, map_location=device, weights_only=False)
    model = LM(ModelConfig(**ck["config"])).to(device)
    model.load_state_dict(ck["model"])
    card = run / "run_card.yaml"
    return model.eval(), (yaml.safe_load(card.read_text()) if card.exists() else None)
