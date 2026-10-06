"""Model ladders: the sizes, budgets and run plans of Module 11, and reading the results back.

A *ladder* is a set of small runs, all with the recipe of the big run, chosen so that a fit through them says
something about the big run before it starts (lesson 11.3). Two kinds of plan:

* :func:`plan_isoflop`: several sizes at each of several compute budgets (Hoffmann et al.'s approach 2 grid);
* :func:`plan_fixed_ratio`: each size at one or more tokens-per-parameter ratios (the OLMo ladder uses
  1×, 2×, 5× and 10× Chinchilla's 20 tokens per parameter, arXiv 2412.04403).

Importing this module adds the ladder sizes to ``frontierlab.model.PRESETS`` (at run time, in this process
only; the shared file is not edited), so every course wrapper (``frontierlab.train.loop``,
``frontierlab.optim.train``, ``frontierlab.datax.train``) accepts ``--preset m11-r3`` once
``frontierlab.scaling`` is imported. ``python -m frontierlab.scaling.train`` does exactly that.

Compute is counted with ``frontierlab.attention.accounting.flops_per_token`` (exact for the model: every
matmul, the attention scores and the tied output head), not with 6·N·D; at the CPU sizes the output head is
most of the compute, which is the point Porian et al. make about Kaplan's fits (lesson 11.1).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from frontierlab.model import PRESETS, ModelConfig
from frontierlab.model.config import param_counts

# CPU rungs (head_dim 32), built for Data-v0 retokenized with a 1,024-token vocabulary (lesson 11.1 explains
# why: at vocabulary 8,192 the tied head is most of every small model and the CPU budgets cannot reach an
# interior iso-FLOP minimum). Shapes: (hidden, layers, heads, kv_heads, swiglu inner). Total parameters at
# vocabulary 1024: 46K, 105K, 164K, 394K, 919K, 2.17M, 4.49M.
CPU_RUNGS = {
    "m11-r1": (32, 1, 1, 1, 96),
    "m11-r2": (48, 2, 2, 1, 128),
    "m11-r3": (64, 2, 2, 1, 192),
    "m11-r4": (96, 3, 3, 1, 256),
    "m11-r5": (128, 4, 4, 2, 384),
    "m11-r6": (192, 5, 6, 2, 512),
    "m11-r7": (256, 6, 8, 2, 704),
}
CPU_VOCAB = 1024
# Main-path rungs (vocabulary 32768, head_dim 64), in addition to pilot-10m/30m/70m and baseline0.
MAIN_RUNGS = {
    "m11-200m": (1024, 14, 16, 4, 2816),
    "m11-350m": (1024, 24, 16, 4, 3584),
    "m11-1b": (1536, 28, 24, 8, 5376),
}


def _cpu(shape):
    C, L, H, KV, I = shape
    def make(vocab_size: int = CPU_VOCAB) -> ModelConfig:
        return ModelConfig(vocab_size=vocab_size, hidden_size=C, num_hidden_layers=L, num_attention_heads=H,
                           num_key_value_heads=KV, head_dim=32, intermediate_size=I, max_position_embeddings=512)
    return make


def _main(shape):
    C, L, H, KV, I = shape
    def make(vocab_size: int = 32768) -> ModelConfig:
        return ModelConfig(vocab_size=vocab_size, hidden_size=C, num_hidden_layers=L, num_attention_heads=H,
                           num_key_value_heads=KV, head_dim=64, intermediate_size=I, max_position_embeddings=4096)
    return make


def register() -> None:
    for name, shape in CPU_RUNGS.items():
        PRESETS.setdefault(name, _cpu(shape))
    for name, shape in MAIN_RUNGS.items():
        PRESETS.setdefault(name, _main(shape))


register()

MAIN_LADDER = ["pilot-10m", "pilot-30m", "pilot-70m", "baseline0", "m11-200m"]


def config(preset: str, vocab_size: int | None = None) -> ModelConfig:
    fn = PRESETS[preset]
    return fn() if vocab_size is None else fn(vocab_size=vocab_size)


def fpt(cfg: ModelConfig, seq: int) -> float:
    """Training FLOPs per token, exact for the course model (forward + backward = 3 × forward)."""
    from frontierlab.attention.accounting import flops_per_token
    return float(flops_per_token(cfg, seq))


def sizes(cfg: ModelConfig) -> dict:
    pc = param_counts(cfg)
    return {"N_nonemb": pc["non_embedding"], "N_total": pc["total"], "embedding": pc["embedding"]}


def plan_isoflop(budgets, presets, *, seq: int, batch: int, vocab_size: int | None = None, grad_accum: int = 1,
                 min_steps: int = 40) -> list[dict]:
    """One run per (budget, preset): steps = round(budget / (FLOPs per token × tokens per step)).

    Runs with fewer than ``min_steps`` steps are dropped (too short for the warmup and schedule to mean
    anything). Each entry has the run's actual FLOPs ``C`` (rounding to whole steps changes it slightly).
    """
    tps = batch * seq * grad_accum
    out = []
    for budget in budgets:
        for p in presets:
            cfg = config(p, vocab_size)
            f = fpt(cfg, seq)
            steps = int(round(budget / (f * tps)))
            if steps < min_steps:
                continue
            out.append({"budget": float(budget), "preset": p, "steps": steps, "tokens": steps * tps,
                        "C": f * steps * tps, "fpt": f, **sizes(cfg)})
    return out


def plan_fixed_ratio(presets, ratios, *, seq: int, batch: int, vocab_size: int | None = None, grad_accum: int = 1,
                     key_N: str = "N_total") -> list[dict]:
    """Each preset at each ratio of tokens per parameter (``key_N`` parameters). The OLMo ladder's 1×, 2×, 5×, 10×
    of Chinchilla are ratios 20, 40, 100, 200."""
    tps = batch * seq * grad_accum
    out = []
    for p in presets:
        cfg = config(p, vocab_size)
        s = sizes(cfg)
        for r in ratios:
            tokens = r * s[key_N]
            steps = max(1, int(round(tokens / tps)))
            out.append({"preset": p, "ratio": r, "steps": steps, "tokens": steps * tps, "C": fpt(cfg, seq) * steps * tps,
                        "fpt": fpt(cfg, seq), **s})
    return out


def projected_gpu_hours(C: float, peak: float = 989e12, mfu: float = 0.30) -> float:
    """Plan section 12.1: GPU-hours = training FLOPs ÷ (peak × MFU). PROJECTED until a pilot measures the MFU."""
    return C / (peak * mfu) / 3600.0


def read_run(run_dir) -> dict:
    """Final and per-evaluation validation losses, tokens, FLOPs and sizes of one finished run."""
    from frontierlab.metrics.jsonl import read_jsonl
    from frontierlab.runcard import read_run_card
    run_dir = Path(run_dir)
    card = read_run_card(run_dir)
    rows = read_jsonl(run_dir / "metrics.jsonl")
    val = [r for r in rows if r.get("split") == "val"]
    train = [r for r in rows if r.get("split") == "train"]
    cfg = ModelConfig(**card["config"])
    steps = int(card["budget"]["steps"])
    tokens = float(card["budget"]["tokens"])
    last = {}
    for r in val:                                   # keep the last value per step (a resumed run may repeat one)
        last[int(r["step"])] = float(r["loss"])
    curve = sorted(last.items())
    return {"run": run_dir.name, "steps": steps, "tokens": tokens, "D": tokens, "C": float(card["budget"]["train_flops"]),
            "loss": curve[-1][1] if curve and curve[-1][0] == steps else math.nan, "val_curve": curve,
            "train_curve": [(int(r["step"]), float(r["loss"])) for r in train],
            "grad_norms": [(int(r["step"]), float(r["grad_norm"])) for r in train],
            "finished": bool(curve) and curve[-1][0] == steps, **sizes(cfg)}


def save_json(obj, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
    return path
