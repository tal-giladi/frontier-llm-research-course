"""Lab 19.1: score three proposals, test how robust the ranking is, and run one cheap proxy.

    python labs/module-19/lesson-01/choose_lab.py --part rank        # seconds: your PROPOSALS scored and stress-tested
    python labs/module-19/lesson-01/choose_lab.py --part proxy       # free CPU, about 10 minutes: a 2-width proxy ladder
    python labs/module-19/lesson-01/choose_lab.py --variant main --print

Part rank: your ``PROPOSALS`` (lab.py) scored with your ``score`` and ``p_decisive``; the minimum detectable effect of
each design (``frontierlab.stats.min_detectable_effect``); how often the top proposal stays on top when every estimate
is off by up to 2x and 3x (``frontierlab.research.proposals.rank_robustness``). Written to runs/m19/l191/proposals.md.

Part proxy: the cheap proxy for the capstone claim "QK-Clip vs QK-norm": is the held-out-loss cost of removing
QK-norm at a raised learning rate (1e-2) growing, shrinking or flat with width? Two widths of the toy preset
(64 and 128), two seeds, QK-norm on and off, 200 steps of 16 x 128 tokens; read with your ``proxy_trend``.
Runs go to runs/m19/l191/<variant>/<arm>-w<width>-s<seed>; rerunning skips finished runs.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import time
from pathlib import Path

import numpy as np

from frontierlab.flops import PEAK_BF16, flops_per_token
from frontierlab.labkit import load_path
from frontierlab.metrics import read_jsonl
from frontierlab.model import PRESETS
from frontierlab.optim import mup
from frontierlab.optim import train as optim_train
from frontierlab.research.proposals import CAPSTONE_CLAIMS, Proposal, check_proposals, rank_robustness
from frontierlab.stats import min_detectable_effect

HERE = Path(__file__).resolve().parent
LAB = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
OUT = Path("runs/m19/l191")
LR = 1e-2
ARMS = {"qknorm": ["--qk-norm", "on"], "noqk": ["--qk-norm", "off"]}
VARIANTS = {   # preset, widths, steps, batch, seq, seeds, extra
    "cpu": ("toy", [64, 128], 200, 16, 128, [0, 1],
            ["--eval-every", "200", "--eval-windows", "128", "--ckpt-every", "100", "--log-every", "20"]),
    "main": ("pilot-30m", [256, 512], 2000, 32, 1024, [0, 1],
             ["--dtype", "bf16", "--eval-every", "2000", "--eval-windows", "256", "--ckpt-every", "500",
              "--log-every", "20", "--peak", "H100-SXM"]),
}
ASSUMED_MFU = 0.25


def argv_for(variant, arm, width, seed):
    preset, _, steps, batch, seq, _, extra = VARIANTS[variant]
    run = OUT / variant / f"{arm}-w{width}-s{seed}"
    return ["--run", run.as_posix(), "--preset", preset, "--width", str(width), "--steps", str(steps),
            "--batch", str(batch), "--seq", str(seq), "--lr", f"{LR:g}", "--warmup", str(steps // 10),
            "--seed", str(seed), "--optimizer", "adamw-torch",
            "--question", f"lesson 19.1 proxy ladder, arm {arm}", *ARMS[arm], *extra]


def final_val(run: Path, steps: int):
    p = run / "metrics.jsonl"
    if not p.exists():
        return None
    rows = [r for r in read_jsonl(p) if r["split"] == "val" and r["step"] >= steps]
    return rows[-1]["loss"] if rows else None


def proxy(variant: str) -> None:
    preset, widths, steps, _, _, seeds, _ = VARIANTS[variant]
    t0 = time.perf_counter()
    for seed in seeds:
        for w in widths:
            for arm in ARMS:
                argv = argv_for(variant, arm, w, seed)
                if final_val(Path(argv[1]), steps) is None:
                    optim_train.main(argv)
    print(f"\ntraining: {(time.perf_counter() - t0) / 60:.1f} minutes")
    sizes, effects, diffs = [], [], []
    print(f"{'width':>6s} {'params':>10s} " + " ".join(f"{'s' + str(s) + ' on/off':>16s}" for s in seeds) + f" {'cost of no QK-norm':>19s}")
    for w in widths:
        cfg = mup.width_config(PRESETS[preset](vocab_size=8192 if variant == "cpu" else 32768), w)
        from frontierlab.model import param_counts
        n = param_counts(cfg)["total"]
        d, cells = [], []
        for s in seeds:
            on = final_val(OUT / variant / f"qknorm-w{w}-s{s}", steps)
            off = final_val(OUT / variant / f"noqk-w{w}-s{s}", steps)
            cells.append(f"{on:.3f}/{off:.3f}")
            d.append(off - on)
        diffs.append(d)
        sizes.append(n)
        effects.append(float(np.mean(d)))
        print(f"{w:6d} {n:10,d} " + " ".join(f"{c:>16s}" for c in cells) + f" {effects[-1]:+19.4f}")
    resid = np.concatenate([np.asarray(d) - np.mean(d) for d in diffs])
    noise = float(np.sqrt((resid ** 2).sum() / max(1, resid.size - len(diffs))))
    print(f"seed noise of one difference (pooled std): {noise:.4f}")
    verdict = LAB.proxy_trend(sizes, effects, noise)
    print(f"proxy_trend: {verdict}")
    (OUT / variant).mkdir(parents=True, exist_ok=True)
    (OUT / variant / "proxy.json").write_text(json.dumps({"sizes": sizes, "effects": effects, "diffs": diffs,
                                                          "noise": noise, "verdict": verdict}, indent=1))


def rank() -> None:
    props, rows = [], []
    for d in LAB.PROPOSALS:
        pd = LAB.p_decisive(d["effect"], d["seed_std"], d["n_seeds"], d["p_transfer"])
        p = Proposal(d["claim_id"], d["question"], d["decision"], d["value"], pd, d["cost_gpu_h"], d["proxy"], d["kill"])
        props.append(p)
        mde = min_detectable_effect(d["seed_std"], d["n_seeds"])
        rows.append((p, LAB.score(p.value, pd, p.cost_gpu_h), mde, d))
    problems = check_proposals(props)
    lines = ["# Three proposals, ranked", "", "| # | claim | value | effect | MDE | p_transfer | p_decisive | GPU-h | score |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, (p, sc, mde, d) in enumerate(rows, 1):
        lines.append(f"| {i} | {p.claim_id} | {p.value:g} | {d['effect']:g} | {mde:.3g} | {d['p_transfer']:g} | "
                     f"{p.p_decisive:.3f} | {p.cost_gpu_h:g} | {sc:.4f} |")
    lines.append("")
    for f in (2.0, 3.0):
        lines.append(f"Top proposal stays on top when every estimate is off by up to {f:g}x: "
                     f"{rank_robustness(props, factor=f):.0%} of draws.")
    lines.append("")
    for p, _, mde, d in rows:
        c = CAPSTONE_CLAIMS.get(p.claim_id)
        lines.append(f"- **{p.claim_id}**: {p.question} Decision: {p.decision}. Proxy: {p.proxy}. Kill: {p.kill}."
                     + (f" Underpowered: the effect {d['effect']:g} is below the MDE {mde:.3g}." if abs(d['effect']) < mde else "")
                     + (f" Course lab cost: {'-'.join(sorted({f'{x:g}' for x in c['lab_cost_gpu_h']}, key=float))} GPU-h ({c['cost_source']})."
                        if c else ""))
    if problems:
        lines += ["", "Problems:"] + [f"- {x}" for x in problems]
    text = "\n".join(lines) + "\n"
    print(text)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "proposals.md").write_text(text, encoding="utf-8")


def print_commands(variant: str) -> None:
    preset, widths, steps, batch, seq, seeds, _ = VARIANTS[variant]
    total = 0.0
    for w in widths:
        cfg = mup.width_config(PRESETS[preset](), w)
        total += len(ARMS) * len(seeds) * flops_per_token(cfg, seq) * steps * batch * seq
    hours = total / (PEAK_BF16["H100-SXM"] * ASSUMED_MFU) / 3600
    print(f"# PROJECTED: {total:.3g} FLOPs / (989e12 x {ASSUMED_MFU} MFU) = {hours:.2f} H100-hours (MFU assumed)")
    for seed in seeds:
        for w in widths:
            for arm in ARMS:
                print("python -m frontierlab.optim.train " + " ".join(shlex.quote(x) for x in argv_for(variant, arm, w, seed)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["rank", "proxy", "all"], default="all")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    if a.print:
        print_commands(a.variant)
        return
    if a.part in ("rank", "all"):
        rank()
    if a.part in ("proxy", "all"):
        proxy(a.variant)


if __name__ == "__main__":
    main()
