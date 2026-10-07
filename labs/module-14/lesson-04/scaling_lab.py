"""Lab 14.4: fitting RL compute curves, why a short run cannot identify the asymptote, and pass@k.

    python labs/module-14/lesson-04/scaling_lab.py                 # free CPU, about 10 minutes
    python labs/module-14/lesson-04/scaling_lab.py --part published
    python labs/module-14/lesson-04/scaling_lab.py --variant main --print

Part ``published`` — a curve with ScaleRL's reported 8B fit (A = 0.645, B = 1.70; appendix fit-window study)
and its reported run-to-run noise (A within +-0.015 over 3 runs, so per-point noise 0.005 here), sampled at
198 evenly spaced checkpoints (every 500 GPU-hours) from 1.5k to 100k GPU-hours. C_mid and R_0 are not given in the paper's text:
this lab uses R_0 = 0.30 and C_mid = 8,000 GPU-hours as stand-ins (labelled; replace them with values you
digitise from the paper's Figure 1). The sigmoid is fitted on several windows, as ScaleRL's appendix does, and
the 95% profile interval of A is reported for each.

Part ``own`` — your own short run: GRPO from the Module 12 SFT start for 200 steps, held-out accuracy every 10
steps, compute measured in sampled responses. The same fit and profile on the first 50, 100 and 200 steps.

Part ``passk`` — pass@k for k = 1 ... 256 from n = 256 samples per problem on 200 held-out problems, for the
SFT start, the 200-step RL policy and a random-reward control (GRPO, same settings, Bernoulli(0.5) reward):
the curves, paired differences at k = 1 and k = 256, the crossover, and which problems each model ever solves.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from frontierlab.labkit import load_path
from frontierlab.metrics.jsonl import read_jsonl
from frontierlab.posttrain import rl
from frontierlab.posttrain.sft import ensure_sft, load_policy
from frontierlab.posttrain.tasks import problems_from, split_problems
from frontierlab.rlscale import curves, passk, runner

HERE = Path(__file__).resolve().parent
ROOT = Path("runs/m14/l144")
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def show_profile(label, C, y, R0):
    p = curves.profile_asymptote(C, y, R0)
    f = p["fit"]
    lo, hi = p["interval"]
    bound = f"[{lo:.3f}, {hi:.3f}]" + ("" if p["bounded_above"] else "  <- runs to A = 1: not identified")
    print(f"  {label:26s} n={f['n']:3d}  A {f['A']:.3f}  B {f['B']:.2f}  C_mid {f['C_mid']:10.4g}   95% profile A {bound}")
    return p


def published(lab):
    rng = np.random.default_rng(0)
    C = np.linspace(1.5e3, 1e5, 198)
    y = lab.sigmoid_curve(C, 0.645, 1.70, 8e3, 0.30) + rng.normal(0, 0.005, C.size)
    print("Reconstructed ScaleRL-like curve: A = 0.645, B = 1.70 (published); C_mid = 8,000, R_0 = 0.30 (stand-ins)")
    for lo, hi in ((1.5e3, 1e5), (1.5e3, 5e4), (5e3, 5e4), (1.5e3, 1.6e4), (1.5e3, 8e3), (1.5e3, 4e3)):
        Cw, yw = curves.window(C, y, lo, hi)
        show_profile(f"window {lo / 1e3:g}k-{hi / 1e3:g}k GPU-h", Cw, yw, 0.30)


def own_run(sft):
    run = ROOT / "grpo-200"
    base = rl.RLConfig(init=str(sft), steps=200, eval_every=10, eval_n=300)
    cfg = runner.objective_config(base, "grpo", seed=0, run=str(run))
    if not (run / "policy.pt").exists():
        runner.train_objective(cfg, "grpo")
    ctl = ROOT / "control-random-200"
    if not (ctl / "policy.pt").exists():
        runner.train_objective(runner.objective_config(base, "grpo", seed=0, run=str(ctl)), "grpo", control="random")
    return run, ctl


def own(lab, sft):
    run, _ = own_run(sft)
    _, held = split_problems(2)
    probs = problems_from(held, ".", 2, "+")[:300]
    R0 = rl.evaluate(load_policy(sft), probs, 8, 1.0, torch.Generator().manual_seed(10_000))["sampled_acc"]
    ev = [r for r in read_jsonl(run / "metrics.jsonl") if r["split"] == "eval"]
    C = np.array([r["step"] * 16 * 8 for r in ev], float)
    y = np.array([r["sampled_acc"] for r in ev])
    print(f"\nYour run: GRPO, 200 steps, R_0 = {R0:.3f} (SFT start, same evaluation), compute = sampled responses")
    print("  accuracy every 10 steps: " + " ".join(f"{v:.2f}" for v in y))
    for steps in (50, 100, 200):
        k = C <= steps * 128
        show_profile(f"first {steps} steps", C[k], y[k], R0)


def passk_part(lab, sft):
    run, ctl = own_run(sft)
    _, held = split_problems(2)
    probs = problems_from(held, ".", 2, "+")[:200]
    n = 256
    counts = {}
    for name, path in (("SFT start", sft), ("GRPO 200 steps", run / "policy.pt"), ("random reward", ctl / "policy.pt")):
        counts[name] = passk.toy_counts(load_policy(path), probs, n)
    curves_ = {k: lab.pass_at_k_curve(v, n, KS) for k, v in counts.items()}
    print(f"\npass@k on 200 held-out problems, n = {n} samples each")
    print(f"  {'k':>16s} " + " ".join(f"{k:6d}" for k in KS))
    for name, c in curves_.items():
        print(f"  {name:>16s} " + " ".join(f"{v:6.3f}" for v in c))
    base = counts["SFT start"]
    for name in ("GRPO 200 steps", "random reward"):
        d = passk.paired_curve_diff(counts[name], base, n, [1, 256])
        cr = lab.crossover(curves_[name], curves_["SFT start"], KS)
        print(f"  {name} - SFT: pass@1 {d[0]['diff']:+.3f} [{d[0]['ci'][0]:+.3f}, {d[0]['ci'][1]:+.3f}], "
              f"pass@256 {d[1]['diff']:+.3f} [{d[1]['ci'][0]:+.3f}, {d[1]['ci'][1]:+.3f}], crossover k = {cr}; "
              f"solved ever {passk.solved_sets(counts[name], base)} (a = {name}, b = SFT)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["all", "published", "own", "passk"], default="all")
    ap.add_argument("--variant", choices=["cpu", "main", "t4"], default="cpu")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    if a.variant != "cpu":
        extra = "" if a.variant == "main" else (" --model Qwen/Qwen3-0.6B-Base --revision "
                                                "da87bfb608c14b7cf20ba1ce41287e8de496c0cd --prompts 8 --max-new 256")
        print(f"python -m frontierlab.rlscale.hf_rl --objective cispo --steps 300 --eval-every 25 --eval-samples 64 "
              f"--eval-n 200 --run runs/m14/l144-{a.variant}/cispo-300{extra}")
        print(f"python -m frontierlab.rlscale.hf_rl --objective cispo --control random --steps 300 --eval-every 100 "
              f"--eval-samples 64 --eval-n 200 --run runs/m14/l144-{a.variant}/random-300{extra}")
        print("# pass@k from the saved per-question counts (eval rows 'counts'); fit the eval curve with "
              "frontierlab.rlscale.curves.profile_asymptote")
        return
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if a.part in ("all", "published"):
        published(lab)
    if a.part in ("all", "own", "passk"):
        sft = ensure_sft("runs/m12/sft")
        if a.part in ("all", "own"):
            own(lab, sft)
        if a.part in ("all", "passk"):
            passk_part(lab, sft)


if __name__ == "__main__":
    main()
