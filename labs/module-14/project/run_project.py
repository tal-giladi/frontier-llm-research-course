"""Module 14 project: a reasoning-RL run with a stability report, Eval v2 retention and a pass@k analysis.

    python labs/module-14/project/run_project.py --objective cispo            # free CPU, about 30 minutes
    python labs/module-14/project/run_project.py --objective cispo --report   # tables only (runs must exist)
    python labs/module-14/project/run_project.py --variant main --objective cispo --print
    python labs/module-14/project/buggy_run.py                                # the debugging task

Free CPU: the chosen objective (your lesson 14.2 decision) and a random-reward control arm (same objective,
Bernoulli(0.5) reward), seeds 0-2, 200 steps from the Module 12 SFT start, 2 epochs x 4 minibatches per batch,
held-out accuracy every 20 steps. Then, per seed:

1. stability report: the pre-stated metrics and flags of lesson 14.2;
2. Eval Suite v2 against the SFT start with the guards below;
3. pass@k for k = 1 ... 256 from n = 256 samples on 200 held-out problems, against the SFT start;
4. the profile interval of the asymptote A from the run's own curve (lesson 14.4).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from frontierlab.evals.suite_v2 import compare as v2_compare
from frontierlab.evals.suite_v2.toy import run_suite
from frontierlab.metrics.jsonl import read_jsonl
from frontierlab.posttrain import rl
from frontierlab.posttrain.arms import t_interval
from frontierlab.posttrain.sft import ensure_sft, load_policy
from frontierlab.posttrain.tasks import problems_from, split_problems
from frontierlab.rlscale import curves, passk, runner
from frontierlab.rlscale.stability import run_stability

ROOT = Path("runs/m14/project")
SEEDS = [0, 1, 2]
GUARDS = {"sub_greedy": 0.02, "if_strict": 0.02, "if_correct": 0.02, "sft_nll": 0.02}
THRESHOLDS = {"collapse_drawdown": 0.10, "grad_spikes": 3, "entropy_drop": 0.25, "ratio_max": 50.0}
KS = [1, 4, 16, 64, 256]


def flags(st):
    out = []
    for name, key, th in (("collapse", "eval_drawdown", "collapse_drawdown"), ("grad_spikes", "grad_spikes", "grad_spikes"),
                          ("entropy_collapse", "entropy_drop", "entropy_drop"), ("ratio_blowup", "ratio_max", "ratio_max")):
        v = st.get(key)
        if v is not None and v == v and v >= THRESHOLDS[th]:
            out.append(name)
    return out or ["-"]


def main_print(objective):
    root = "runs/m14/project-main"
    for arm, ctl in ((objective, ""), (f"{objective}-random", " --control random")):
        for s in (0, 1):
            print(f"python -m frontierlab.rlscale.hf_rl --objective {objective}{ctl} --seed {s} --steps 300 "
                  f"--eval-every 50 --eval-n 200 --eval-samples 64 --run {root}/{arm}-s{s}")
    for arm in (objective, f"{objective}-random"):
        for s in (0, 1):
            print(f"python labs/module-12/lesson-04/eval_main.py --model {root}/{arm}-s{s}/policy "
                  f"--base Qwen/Qwen3-1.7B-Base --out {root}/eval/{arm}-s{s}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objective", default="cispo", choices=["grpo", "dapo", "drgrpo", "gspo", "cispo"])
    ap.add_argument("--lr", type=float, default=3e-4, help="your lesson 14.2 tuned learning rate")
    ap.add_argument("--variant", choices=["cpu", "main"], default="cpu")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if a.variant == "main":
        main_print(a.objective)
        return
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    sft = ensure_sft("runs/m12/sft")
    base = rl.RLConfig(init=str(sft), steps=200, epochs=2, minibatches=4, eval_every=20, eval_n=300, lr=a.lr)
    arms = {a.objective: {"objective": a.objective},
            f"{a.objective}-random": {"objective": a.objective, "control": "random"}}
    if not a.report:
        runner.run_arms(base, arms, SEEDS, ROOT)
    sft_suite = run_suite(load_policy(sft))
    _, held = split_problems(2)
    probs = problems_from(held, ".", 2, "+")[:200]
    sft_counts = passk.toy_counts(load_policy(sft), probs, 256)
    R0 = rl.evaluate(load_policy(sft), problems_from(held, ".", 2, "+")[:300], 8, 1.0,
                     torch.Generator().manual_seed(10_000))["sampled_acc"]
    print(f"SFT start: held-out sampled accuracy {R0:.3f}; pass@k " +
          " ".join(f"@{k} {v:.3f}" for k, v in zip(KS, passk.curve(sft_counts, 256, KS))))
    finals = {}
    for arm in arms:
        print(f"\n=== {arm}")
        finals[arm] = []
        for s in SEEDS:
            run = ROOT / f"{arm}-s{s}"
            ev = [r for r in read_jsonl(run / "metrics.jsonl") if r["split"] == "eval"]
            finals[arm].append(ev[-1]["sampled_acc"])
            st = run_stability(run)
            cache = run / "eval_v2.json"
            if not cache.exists():
                cache.write_text(json.dumps(run_suite(load_policy(run / "policy.pt"))))
            cmp = v2_compare(sft_suite, json.loads(cache.read_text()), guards=GUARDS)
            pc = run / "passk_counts.json"
            if not pc.exists():
                pc.write_text(json.dumps(passk.toy_counts(load_policy(run / "policy.pt"), probs, 256).tolist()))
            counts = np.array(json.loads(pc.read_text()))
            d = passk.paired_curve_diff(counts, sft_counts, 256, [1, 256])
            C = np.array([r["step"] * 128 for r in ev], float)
            y = np.array([r["sampled_acc"] for r in ev])
            rising = y[-1] > R0
            prof = curves.profile_asymptote(C, y, R0) if rising else None
            lo, hi = prof["interval"] if rising else (float("nan"), float("nan"))
            regressed = [k for k, v in cmp.items() if not k.startswith("_") and v["verdict"] == "regressed"]
            print(f" seed {s}: final acc {y[-1]:.3f}; stability: entropy drop {st['entropy_drop']:+.3f}, grad p99 "
                  f"{st['grad_norm_p99']:.2f}, spikes {st['grad_spikes']}, ratio max {st['ratio_max']:.1f}, clip "
                  f"{st['clip_frac']:.3f}, KL {st['kl_final']:.3f}, drawdown {st['eval_drawdown']:.3f}, flags {','.join(flags(st))}")
            print(f"         Eval v2 vs SFT: {'PASS' if cmp['_passes'] else 'FAIL'}" +
                  (f" (regressed: {', '.join(regressed)})" if regressed else "") +
                  "; " + ", ".join(f"{k} {v['diff']:+.3f}" for k, v in cmp.items() if not k.startswith("_")))
            print(f"         pass@1 vs SFT {d[0]['diff']:+.3f} [{d[0]['ci'][0]:+.3f}, {d[0]['ci'][1]:+.3f}], "
                  f"pass@256 {d[1]['diff']:+.3f} [{d[1]['ci'][0]:+.3f}, {d[1]['ci'][1]:+.3f}], "
                  f"crossover k = {passk.crossover(passk.curve(counts, 256, KS), passk.curve(sft_counts, 256, KS), KS)}; "
                  f"solved ever: {passk.solved_sets(counts, sft_counts)}")
            if rising:
                print(f"         asymptote from this run: A {prof['fit']['A']:.3f}, 95% profile [{lo:.3f}, {hi:.3f}]"
                      + ("" if prof["bounded_above"] else " (not bounded above)"))
            else:
                print("         asymptote: the curve ended below R_0, so there is no rising curve to fit")
    o, c = finals[a.objective], finals[f"{a.objective}-random"]
    m, lo, hi = t_interval(np.array(o) - np.array(c))
    print(f"\n{a.objective} - random control, final accuracy, paired by seed: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
