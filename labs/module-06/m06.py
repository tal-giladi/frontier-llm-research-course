"""Shared helpers for the Module 6 labs and project: the arms, the variants, training, evaluation, comparison.

Lab scripts add ``labs/module-06`` to ``sys.path`` and ``import m06``. Every run goes through
``python -m frontierlab.blocks.train`` (the unmodified course loop plus the Module 6 wrapper), so exact
resume holds: rerunning a script skips finished runs and continues interrupted ones.

Runs live in ``runs/m06/<variant>/<arm>/s<seed>`` and are shared between lessons: lesson 06.3's factorial and
the Lineage-F project reuse the b0, MTP and mHC runs of lessons 06.1 and 06.2 when the variant, arm and seed
match (same command line, so the same run). Arms are defined relative to the preset and the training
length T, so a name means the same design at every size.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from frontierlab.blocks import accounting
from frontierlab.blocks import train as blocks_train
from frontierlab.data.loader import TokenData
from frontierlab.evals.heldout import window_losses
from frontierlab.metrics import read_jsonl
from frontierlab.model import PRESETS
from frontierlab.stats import min_detectable_effect, paired_bootstrap, summary

ROOT = Path("runs/m06")

# preset, steps, batch, seq, lr, extra loop args, eval windows x length
VARIANTS = {
    "cpu": dict(preset="toy", steps=200, batch=16, seq=128, lr=1.5e-3,
                loop=["--eval-every", "200", "--ckpt-every", "100", "--eval-windows", "32", "--log-every", "10"],
                eval_n=256, eval_T=128),
    "t4": dict(preset="pilot-10m", steps=2000, batch=32, seq=512, lr=1.5e-3,
               loop=["--dtype", "fp32", "--eval-every", "500", "--ckpt-every", "250", "--log-every", "10"],
               eval_n=256, eval_T=512),
    "main": dict(preset="pilot-30m", steps=4000, batch=64, seq=1024, lr=1.5e-3,
                 loop=["--dtype", "bf16", "--eval-every", "500", "--ckpt-every", "500", "--log-every", "10",
                       "--peak", "H100-SXM"], eval_n=256, eval_T=1024),
    "main70": dict(preset="pilot-70m", steps=4000, batch=64, seq=1024, lr=1.5e-3,
                   loop=["--dtype", "bf16", "--eval-every", "500", "--ckpt-every", "500", "--log-every", "10",
                         "--peak", "H100-SXM"], eval_n=256, eval_T=1024),
    # the Lineage-F project's main path: Baseline-0's shape and budget (Module 1 project: 9,500 x 32 x 8 x 1,024
    # tokens). lr=None: pass Baseline-0's tuned learning rate with --lr (the project refuses to guess it).
    "lineage": dict(preset="baseline0", steps=9500, batch=32, seq=1024, lr=None,
                    loop=["--grad-accum", "8", "--dtype", "bf16", "--eval-every", "500", "--ckpt-every", "250",
                          "--log-every", "20", "--peak", "H100-SXM", "--max-minutes", "600"], eval_n=256, eval_T=1024),
}


def mla_flags(preset: str) -> list[str]:
    c = PRESETS[preset]()
    return ["--attention", "mla", "--extra",
            json.dumps({"kv_lora_rank": c.num_key_value_heads * c.head_dim, "qk_rope_head_dim": c.head_dim // 2})]


def local_global_flags(preset: str, T: int) -> list[str]:
    L = PRESETS[preset]().num_hidden_layers
    return ["--attention", "local_global", "--extra", json.dumps({"global_every": 6 if L >= 6 else 4,
                                                                  "window": max(16, T // 4)})]


MTP_DS = ["--mtp", "deepseek", "--mtp-depth", "1", "--mtp-schedule", "deepseek"]
MTP_META = ["--mtp", "meta", "--mtp-depth", "1"]
MHC = ["--residual", "mhc", "--streams", "4"]
HC = ["--residual", "hc", "--streams", "4"]
MOE = ["--ffn", "moe", "--moe-experts", "8", "--moe-top-k", "2", "--moe-shared", "1", "--moe-first-dense", "1"]


def arm_flags(arm: str, preset: str, T: int) -> list[str]:
    """Wrapper flags of an arm. Combined arms join component names with '+' (e.g. 'mla+mtp-ds+mhc+moe')."""
    parts = []
    for comp in arm.split("+"):
        if comp == "b0":
            continue
        elif comp == "mtp-ds":
            parts += MTP_DS
        elif comp == "mtp-meta":
            parts += MTP_META
        elif comp == "mtp-meta-matched":         # Meta's parameter matching: one trunk layer fewer
            parts += MTP_META + ["--cfg", f"num_hidden_layers={PRESETS[preset]().num_hidden_layers - 1}"]
        elif comp == "hc":
            parts += HC
        elif comp == "mhc":
            parts += MHC
        elif comp == "moe":
            parts += MOE
        elif re.fullmatch(r"moe\d+", comp):          # moe6: the MoE arm with 6 routed experts instead of 8
            parts += MOE[:3] + [comp[3:]] + MOE[4:]
        elif re.fullmatch(r"engram\d+", comp):       # engram1465: Engram at layer 1, tables of about 1,465 rows
            parts += ["--engram-layers", "1", "--engram-table", comp[6:], "--engram-heads", "2", "--engram-max-n", "3"]
        elif comp == "diffusion":                     # LLaDA-style masked-diffusion objective, bidirectional attention
            parts += ["--objective", "diffusion"]
        elif comp == "matformer":
            parts += ["--ffn", "matformer"]
        elif re.fullmatch(r"ffn\d+", comp):          # ffn48: Baseline-0 with SwiGLU width 48
            parts += ["--cfg", f"intermediate_size={comp[3:]}"]
        elif comp == "mla":
            parts += mla_flags(preset)
        elif comp == "local-global":
            parts += local_global_flags(preset, T)
        else:
            raise KeyError(f"unknown arm component {comp!r}")
    return parts


def arm_config(arm: str, variant: str, vocab: int):
    """The ModelConfig the wrapper would build for this arm (for accounting without training)."""
    v = VARIANTS[variant]
    flags = arm_flags(arm, v["preset"], v["seq"])
    mine, rest = blocks_train.build_parser().parse_known_args(flags)
    from frontierlab.train import loop
    a = loop.build_parser().parse_args(["--run", "x", "--preset", v["preset"], *rest])
    return blocks_train.model_config(mine, a, vocab)


def vocab_size(data: str | None = None) -> int:
    return TokenData("train", **({"root": data} if data else {})).meta["vocab_size"]


def run_dir(variant: str, arm: str, seed: int, lr: float | None = None, steps: int | None = None, root=ROOT) -> Path:
    name = arm if lr is None else f"{arm}@lr{lr:g}"
    if steps is not None:
        name += f"@{steps}steps"
    return Path(root) / variant / name / f"s{seed}"


def finished(run: Path, steps: int) -> bool:
    log = run / "metrics.jsonl"
    if not log.exists():
        return False
    rows = [r for r in read_jsonl(log) if r["split"] == "train"]
    return bool(rows) and rows[-1]["step"] >= steps


def train_arm(variant: str, arm: str, seed: int, *, lr: float | None = None, steps: int | None = None,
              extra: list[str] | None = None, root=ROOT, question: str = "", max_minutes: float | None = None,
              device: str | None = None, data: str | None = None) -> Path:
    """Train one arm unless it already finished (exact resume otherwise). Returns the run directory."""
    v = VARIANTS[variant]
    n_steps = steps or v["steps"]
    run = run_dir(variant, arm, seed, lr, steps, root)
    if finished(run, n_steps):
        print(f"{run}: finished, skipping")
        return run
    argv = ["--run", str(run), "--preset", v["preset"], "--steps", str(n_steps), "--batch", str(v["batch"]),
            "--seq", str(v["seq"]), "--lr", str(_lr(lr, v)), "--seed", str(seed), "--blocks-log",
            "--question", question or f"Module 6: arm {arm}", *v["loop"], *arm_flags(arm, v["preset"], v["seq"]),
            *(extra or [])]
    if any(c in ("hc", "mhc") for c in arm.split("+")):
        argv += ["--hyper-every", "20"]               # Amax gains every 20 steps (logging only)
    if max_minutes:
        argv += ["--max-minutes", str(max_minutes)]
    if device:
        argv += ["--device", device]
    if data:
        argv += ["--data", data]
    print(f"\n=== {run}")
    t0 = time.perf_counter()
    blocks_train.main(argv)
    dt = time.perf_counter() - t0
    with open(run / "wallclock.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"seconds": round(dt, 2), "time": round(time.time(), 1)}) + "\n")
    return run


def _lr(lr, v):
    if lr is None and v["lr"] is None:
        raise SystemExit("this variant needs --lr (Baseline-0's tuned learning rate from the Module 1 project)")
    return lr or v["lr"]


def wallclock(run: Path) -> float:
    p = Path(run) / "wallclock.jsonl"
    return sum(r["seconds"] for r in read_jsonl(p)) if p.exists() else float("nan")


def eval_losses(run: Path, n: int = 256, T: int = 128, device: str = "cpu", data: str | None = None,
                split: str = "val") -> list[float]:
    """Per-window held-out next-token loss of the run's final checkpoint, cached in <run>/eval_<n>x<T>[_test].json.

    ``split="val"`` while developing and selecting; ``split="test"`` once, for the pre-registered final comparison
    (lesson 01.3)."""
    run = Path(run)
    cache = run / f"eval_{n}x{T}{'' if split == 'val' else '_' + split}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    model = blocks_train.load_model(run / "checkpoint.pt", map_location=device).to(device)
    val = TokenData(split, **({"root": data} if data else {}))
    losses = window_losses(model, val, n, T, device=device)
    cache.write_text(json.dumps(losses))
    return losses


def seed_mean_losses(runs: list[Path], **kw) -> np.ndarray:
    """Per-window losses averaged over seed runs (so a paired bootstrap over windows uses all seeds)."""
    return np.mean([eval_losses(r, **kw) for r in runs], axis=0)


def compare(a, b) -> dict:
    """Paired by window: mean of a − b with a 95% bootstrap interval (lesson 01.4)."""
    return paired_bootstrap(a, b)


def fmt(r: dict) -> str:
    return f"{r['mean_diff']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"


def noise_floor(per_seed_means: list[float], n_per_arm: int) -> dict:
    s = summary(per_seed_means)
    return {"seed_std": s["std"], "mde": min_detectable_effect(s["std"], n_per_arm) if s["n"] > 1 else float("nan")}


def blocks_log(run: Path) -> list[dict]:
    p = Path(run) / "blocks.jsonl"
    return read_jsonl(p) if p.exists() else []


def train_rows(run: Path) -> list[dict]:
    return [r for r in read_jsonl(Path(run) / "metrics.jsonl") if r["split"] == "train"]


def flops_per_token(arm: str, variant: str, vocab: int) -> float:
    return accounting.flops_per_token(arm_config(arm, variant, vocab), VARIANTS[variant]["seq"])


def equal_flops_steps(arm: str, base_arm: str, variant: str, vocab: int, base_steps: int | None = None) -> int:
    """Steps for ``base_arm`` whose training FLOPs equal ``arm`` trained the variant's steps."""
    steps = base_steps or VARIANTS[variant]["steps"]
    return max(1, round(steps * flops_per_token(arm, variant, vocab) / flops_per_token(base_arm, variant, vocab)))
