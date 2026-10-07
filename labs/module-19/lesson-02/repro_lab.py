"""Lab 19.2: reproduce one published claim at small scale and write the reproduction record.

    python labs/module-19/lesson-02/repro_lab.py                     # free CPU: train the sweep, then analyse
    python labs/module-19/lesson-02/repro_lab.py --part analyse      # analyse finished runs only
    python labs/module-19/lesson-02/repro_lab.py --variant main --print   # GPU commands and PROJECTED cost

The claim (Wortsman et al., arXiv 2309.14322, Figure 1 caption and section 3.1.1): "Qk-layernorm reduces LR
sensitivity" — models without it diverge or degrade at high learning rates through attention-logit growth, and the
instability "also appear[s] in small models when training at high learning rates".

The sweep: two arms (QK-norm on, QK-norm off) x learning rates x seeds, everything else as close to the paper's
section 2.1 as is cheap to match (AdamW b2 0.95, eps 1e-8, clipping 1.0, z-loss 1e-4, independent weight decay 1e-4,
5% warm-up then cosine). Per arm and seed, the learning-rate sensitivity of the sweep (your ``lr_sensitivity``);
per seed, the effect "sensitivity without QK-norm minus with" (your ``seed_effects``); the decision against the
tolerance stated in your ``lab.py`` before the runs (your ``decide``); and the record with your ``deviations()``,
validated by ``frontierlab.research.reproduce.validate`` and written to runs/m19/l192/<variant>/record.md.

Runs go to runs/m19/l192/<variant>/<arm>-lr<lr>-s<seed>. Rerunning skips finished runs and resumes interrupted ones.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
import time
from datetime import date
from pathlib import Path

from frontierlab.flops import PEAK_BF16, flops_per_token
from frontierlab.labkit import load_path
from frontierlab.metrics import read_jsonl
from frontierlab.model import PRESETS
from frontierlab.optim import train as optim_train
from frontierlab.research.reproduce import Record, render, validate

HERE = Path(__file__).resolve().parent
LAB = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))

CLAIM = ("Qk-layernorm reduces LR sensitivity (Figure 1 caption): models without it degrade or diverge at high "
         "learning rates through attention-logit growth, also in small models (section 3.1.1).")
SOURCE = "Wortsman et al., Small-scale proxies for large-scale Transformer training instabilities, Figure 1, section 3.1.1"
URL = "https://arxiv.org/abs/2309.14322"

VARIANTS = {   # preset, steps, batch, seq, learning rates, seeds, extra loop flags
    "cpu": ("toy", 300, 16, 128, [3e-3, 1e-2, 3e-2], [0, 1, 2],
            ["--eval-every", "300", "--eval-windows", "128", "--ckpt-every", "100", "--log-every", "20",
             "--stability-every", "5"]),
    "t4": ("pilot-10m", 1000, 16, 256, [1e-3, 3e-3, 1e-2, 3e-2, 1e-1], [0, 1, 2],
           ["--dtype", "fp32", "--eval-every", "1000", "--eval-windows", "128", "--ckpt-every", "250",
            "--log-every", "20", "--stability-every", "10"]),
    "main": ("pilot-30m", 4000, 64, 1024, [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1], [0, 1, 2],
             ["--dtype", "bf16", "--eval-every", "4000", "--eval-windows", "256", "--ckpt-every", "500",
              "--log-every", "20", "--stability-every", "10", "--peak", "H100-SXM"]),
    "main70": ("pilot-70m", 4000, 64, 1024, [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1], [0, 1, 2],
               ["--dtype", "bf16", "--eval-every", "4000", "--eval-windows", "256", "--ckpt-every", "500",
                "--log-every", "20", "--stability-every", "10", "--peak", "H100-SXM"]),
}
ARMS = {"qknorm": ["--qk-norm", "on"], "noqk": ["--qk-norm", "off"]}
INDEPENDENT_DECAY = 1e-4      # Wortsman et al. section 2.1
Z_LOSS = 1e-4
ASSUMED_MFU = 0.25            # PROJECTED until the Module 19 pilot measures it


def run_dir(root: Path, arm: str, lr: float, seed: int) -> Path:
    return root / f"{arm}-lr{lr:g}-s{seed}"


def argv_for(variant: str, root: Path, arm: str, lr: float, seed: int) -> list[str]:
    preset, steps, batch, seq, _, _, extra = VARIANTS[variant]
    return ["--run", run_dir(root, arm, lr, seed).as_posix(), "--preset", preset, "--steps", str(steps),
            "--batch", str(batch), "--seq", str(seq), "--lr", f"{lr:g}", "--seed", str(seed),
            "--warmup", str(max(1, steps // 20)), "--weight-decay", f"{INDEPENDENT_DECAY / lr:.6g}",
            "--optimizer", "adamw", "--z-loss", f"{Z_LOSS:g}", "--stability-log",
            "--question", f"lesson 19.2: Wortsman et al. LR sensitivity, arm {arm}", *ARMS[arm], *extra]


def finished(run: Path, steps: int) -> bool:
    log = run / "metrics.jsonl"
    if not log.exists():
        return False
    rows = [r for r in read_jsonl(log) if r["split"] == "val"]
    return bool(rows) and rows[-1]["step"] >= steps


def train(variant: str, root: Path) -> None:
    _, steps, _, _, lrs, seeds, _ = VARIANTS[variant]
    for seed in seeds:                       # seed-major: an interrupted sweep still has complete seeds
        for lr in lrs:
            for arm in ARMS:
                run = run_dir(root, arm, lr, seed)
                if finished(run, steps):
                    continue
                t0 = time.perf_counter()
                optim_train.main(argv_for(variant, root, arm, lr, seed))
                with open(run / "wallclock.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"seconds": round(time.perf_counter() - t0, 2)}) + "\n")


def collect(variant: str, root: Path) -> dict:
    """{arm: {seed: {lr: {"loss", "max_logit", "seconds"}}}} for finished runs."""
    _, steps, _, _, lrs, seeds, _ = VARIANTS[variant]
    out: dict = {}
    for arm in ARMS:
        for seed in seeds:
            for lr in lrs:
                run = run_dir(root, arm, lr, seed)
                if not finished(run, steps):
                    continue
                val = [r for r in read_jsonl(run / "metrics.jsonl") if r["split"] == "val"][-1]["loss"]
                st = read_jsonl(run / "stability.jsonl") if (run / "stability.jsonl").exists() else []
                mx = max((r["max_logit"] for r in st if r.get("max_logit") is not None), default=float("nan"))
                wc = run / "wallclock.jsonl"
                sec = sum(r["seconds"] for r in read_jsonl(wc)) if wc.exists() else float("nan")
                out.setdefault(arm, {}).setdefault(seed, {})[lr] = {"loss": val, "max_logit": mx, "seconds": sec}
    return out


def analyse(variant: str, root: Path) -> dict | None:
    _, _, _, _, lrs, seeds, _ = VARIANTS[variant]
    res = collect(variant, root)
    vocab = 8192 if variant == "cpu" else 32768
    l0 = math.log(vocab)
    complete = [s for s in seeds if all(len(res.get(a, {}).get(s, {})) == len(lrs) for a in ARMS)]
    print(f"\nFinal held-out loss (l0 = ln {vocab} = {l0:.3f}); max attention logit over the run in brackets")
    print(f"{'arm':8s} {'seed':>4s} " + " ".join(f"{lr:>16g}" for lr in lrs) + f" {'sensitivity':>12s}")
    sens: dict = {a: {} for a in ARMS}
    for arm in ARMS:
        for s in complete:
            row = res[arm][s]
            sens[arm][s] = LAB.lr_sensitivity({lr: row[lr]["loss"] for lr in lrs}, l0)
            cells = " ".join(f"{row[lr]['loss']:7.3f} [{row[lr]['max_logit']:6.1f}]" for lr in lrs)
            print(f"{arm:8s} {s:4d} {cells} {sens[arm][s]:12.4f}")
    if len(complete) < 2:
        print(f"\n{len(complete)} complete seed(s); the decision needs at least 2. Train more seeds.")
        return None
    effects = LAB.seed_effects(sens["noqk"], sens["qknorm"])
    dec = LAB.decide(effects, LAB.TOLERANCE)
    print(f"\nPer-seed effect (sensitivity without QK-norm minus with): {', '.join(f'{e:+.4f}' for e in effects)}")
    print(f"Mean {dec['mean']:+.4f}, 95% t-interval [{dec['ci'][0]:+.4f}, {dec['ci'][1]:+.4f}], "
          f"tolerance {LAB.TOLERANCE:.4f} (stated {LAB.STATED_ON})")
    print(f"Direction: {dec['direction']}. Magnitude: {dec['magnitude']}.")
    top = max(lrs)
    ratios = [res["noqk"][s][top]["max_logit"] / res["qknorm"][s][top]["max_logit"] for s in complete]
    print(f"Secondary (observation, no rule): max-logit ratio without/with QK-norm at lr {top:g}: "
          f"{', '.join(f'{r:.1f}x' for r in ratios)}")
    secs = [v["seconds"] for a in res.values() for sd in a.values() for v in sd.values() if math.isfinite(v["seconds"])]
    if secs:
        print(f"Training time: {sum(secs) / 60:.1f} minutes for {len(secs)} runs ({sum(secs) / len(secs):.0f} s per run)")
    first = min((run_dir(root, a, lr, s) / "run_card.yaml").stat().st_mtime for a in ARMS for s in complete for lr in lrs)
    rec = Record(claim=CLAIM, source=SOURCE, url=URL, tolerance=LAB.TOLERANCE,
                 tolerance_rule=f"2 x the toy recipe's seed std of held-out loss ({LAB.NOISE_FLOOR} nats, lesson 01.4).",
                 stated_on=LAB.STATED_ON, results_on=date.fromtimestamp(first).isoformat(), effects=effects,
                 outcome=dec, deviations=LAB.deviations(),
                 author_contact="not needed: the setup in section 2.1 and the sensitivity definition in section 2.2 "
                                "were complete enough to run; open question for the authors: per-point seed counts in Figure 1.",
                 scope=f"{variant} variant ({PRESETS[VARIANTS[variant][0]]().hidden_size}-wide preset, "
                       f"{VARIANTS[variant][0]}), learning rates {', '.join(f'{x:g}' for x in lrs)}, "
                       f"{VARIANTS[variant][1]} steps; at this scale only, one size, so the paper's "
                       "'sensitivity grows with scale' part is not tested.")
    problems = validate(rec)
    out = root / "record.md"
    out.write_text(render(rec), encoding="utf-8")
    (root / "results.json").write_text(json.dumps({"losses": {a: {str(s): {f"{lr:g}": v for lr, v in d.items()}
                                                                  for s, d in sd.items()} for a, sd in res.items()},
                                                   "sensitivity": {a: {str(s): v for s, v in d.items()} for a, d in sens.items()},
                                                   "decision": dec, "max_logit_ratio_top_lr": ratios}, indent=1))
    print(f"\nRecord written to {out}" + ("" if not problems else " with problems:"))
    for p in problems:
        print(f"  - {p}")
    return dec


def print_commands(variant: str) -> None:
    preset, steps, batch, seq, lrs, seeds, _ = VARIANTS[variant]
    root = Path("runs/m19/l192") / variant
    cfg = PRESETS[preset]()
    tokens = steps * batch * seq
    flops = flops_per_token(cfg, seq) * tokens
    n = len(ARMS) * len(lrs) * len(seeds)
    gpu, peak = ("T4 (fp32 peak 8.1e12 FLOP/s)", 8.1e12) if variant == "t4" else ("H100", PEAK_BF16["H100-SXM"])
    hours = n * flops / (peak * ASSUMED_MFU) / 3600
    print(f"# {variant}: {n} runs of {preset}, {steps} steps x {batch} x {seq} = {tokens:.3g} tokens each")
    print(f"# PROJECTED: {n} x {flops:.3g} FLOPs / ({peak:.3g} x {ASSUMED_MFU} MFU) = {hours:.1f} {gpu} GPU-hours "
          "(MFU assumed until the pilot measures it; the stability log adds a few percent)")
    for seed in seeds:
        for lr in lrs:
            for arm in ARMS:
                print("python -m frontierlab.optim.train " + " ".join(shlex.quote(x) for x in argv_for(variant, root, arm, lr, seed)))
    print(f"LAB_TARGET=solution python labs/module-19/lesson-02/repro_lab.py --variant {variant} --part analyse")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--part", choices=["all", "train", "analyse"], default="all")
    ap.add_argument("--print", action="store_true", help="print the training commands and the PROJECTED cost")
    ap.add_argument("--out", type=Path, default=Path("runs/m19/l192"))
    a = ap.parse_args()
    if a.print:
        print_commands(a.variant)
        return
    root = a.out / a.variant
    if a.part in ("all", "train"):
        train(a.variant, root)
    if a.part in ("all", "analyse"):
        analyse(a.variant, root)


if __name__ == "__main__":
    sys.exit(main())
