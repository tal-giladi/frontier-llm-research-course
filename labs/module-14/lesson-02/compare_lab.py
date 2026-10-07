"""Lab 14.2: a controlled comparison of five RL objectives, with a random-reward control arm.

    python labs/module-14/lesson-02/compare_lab.py                  # free CPU, about 25-35 minutes
    python labs/module-14/lesson-02/compare_lab.py --phase tune     # only the tuning phase
    python labs/module-14/lesson-02/compare_lab.py --variant main --print

Held fixed for every arm: the Module 12 SFT start, 16 prompts x 8 responses per step, temperature 1, the
same prompt stream per seed, KL off (beta = 0), no entropy bonus, 1 epoch x 4 minibatches per batch (the
first minibatch is on-policy, the other three are not), gradient clip 1.0, the Eval v2 pins.
Changed: the objective, with the settings its paper pairs with it (frontierlab.rlscale.objectives.OBJECTIVES).

Phase ``tune``: every objective gets the same budget, 3 learning rates x 1 tuning seed (100) x 60 steps;
your ``select_lr`` picks one from the training pass rate (training prompts only; held-out data unused).
Phase ``compare``: each objective at its selected lr, seeds 0-2, 120 steps; the control arm is GRPO at
GRPO's selected lr with a random reward (Bernoulli 0.5, independent of the response). Every final policy
is scored with Eval Suite v2 against the SFT start. Your ``paired_interval``, ``verdict`` and
``stability_flags`` produce the decision table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from frontierlab.evals.suite_v2 import compare as v2_compare
from frontierlab.evals.suite_v2.toy import run_suite
from frontierlab.labkit import load_path
from frontierlab.metrics.jsonl import read_jsonl
from frontierlab.posttrain import rl
from frontierlab.posttrain.sft import ensure_sft, load_policy
from frontierlab.rlscale import runner
from frontierlab.rlscale.stability import run_stability

HERE = Path(__file__).resolve().parent
ROOT = Path("runs/m14/l142")
NAMES = ["grpo", "dapo", "drgrpo", "gspo", "cispo"]
LRS = [1e-4, 3e-4, 1e-3]
TUNE_SEED, SEEDS = 100, [0, 1, 2]
GUARDS = {"sub_greedy": 0.02, "if_strict": 0.02, "if_correct": 0.02, "sft_nll": 0.02}


def base_cfg(sft, steps):
    return rl.RLConfig(init=str(sft), steps=steps, epochs=1, minibatches=4, eval_every=40, eval_n=300,
                       kl_beta=0.0, kl_place="none")


def last_pass(run, n=25):
    tr = [r for r in read_jsonl(Path(run) / "metrics.jsonl") if r["split"] == "train"]
    return float(np.mean([r["pass"] for r in tr[-n:]]))


def tune(lab, sft):
    chosen = {}
    for name in NAMES:
        res = {}
        for lr in LRS:
            run = ROOT / "tune" / f"{name}-lr{lr:g}"
            cfg = runner.objective_config(base_cfg(sft, 60), name, lr=lr, seed=TUNE_SEED, run=str(run))
            if not (run / "policy.pt").exists():
                runner.train_objective(cfg, name)
            st = run_stability(run)
            res[lr] = {"score": last_pass(run), "collapsed": bool(lab.stability_flags(st)[:1] == ["collapse"])}
        chosen[name] = lab.select_lr(res)
        print(f"tune {name:7s} " + "  ".join(f"lr {lr:g}: {r['score']:.3f}{' (collapsed)' if r['collapsed'] else ''}"
                                             for lr, r in res.items()) + f"   -> {chosen[name]:g}")
    (ROOT / "chosen_lr.json").write_text(json.dumps(chosen))
    return chosen


def compare(lab, sft, chosen):
    arms = {n: {"objective": n, "over": {"lr": chosen[n]}} for n in NAMES}
    arms["control-random"] = {"objective": "grpo", "control": "random", "over": {"lr": chosen["grpo"]}}
    res = runner.run_arms(base_cfg(sft, 120), arms, SEEDS, ROOT / "compare")
    sft_suite = run_suite(load_policy(sft))
    table = {}
    for name in arms:
        finals = [r["final_sampled"] for r in res[name]]
        sts, v2 = [], []
        for s in SEEDS:
            run = ROOT / "compare" / f"{name}-s{s}"
            sts.append(run_stability(run))
            cache = run / "eval_v2.json"
            if not cache.exists():
                cache.write_text(json.dumps(run_suite(load_policy(run / "policy.pt"))))
            cmp = v2_compare(sft_suite, json.loads(cache.read_text()), guards=GUARDS)
            v2.append(cmp)
        table[name] = {"finals": finals, "stability": sts, "v2": v2}
    base, ctl = table["grpo"]["finals"], table["control-random"]["finals"]
    print(f"\nSFT start: held-out sampled accuracy {sft_suite['summary']['add_pass1']:.3f} (Eval v2 add_pass1)")
    print(f"thresholds stated before the runs: {lab.THRESHOLDS}; Eval v2 guards {GUARDS}")
    print(f"\n{'arm':15s} {'final acc per seed':>24s} {'vs grpo (95% t)':>26s} {'vs control (95% t)':>26s}  verdict")
    for name, t in table.items():
        f = t["finals"]
        vb = lab.paired_interval(f, base)
        vc = lab.paired_interval(f, ctl)
        v = "baseline" if name == "grpo" else ("control" if name.startswith("control") else lab.verdict(vb, vc))
        if name == "grpo":
            v = "beats control" if vc[1] > 0 else "below control"
        fmt = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"   # noqa: E731
        print(f"{name:15s} {' '.join(f'{x:.3f}' for x in f):>24s} {fmt(vb) if name != 'grpo' else '':>26s} "
              f"{fmt(vc) if not name.startswith('control') else '':>26s}  {v}")
    print(f"\n{'arm':15s} {'entropy drop':>13s} {'grad p99':>9s} {'spikes':>7s} {'ratio max':>10s} {'clip':>6s} "
          f"{'KL':>6s} {'drawdown':>9s}  flags (per seed) | Eval v2 guard per seed")
    for name, t in table.items():
        S = t["stability"]
        mean = lambda k: float(np.nanmean([s[k] for s in S]))   # noqa: E731
        flags = [",".join(lab.stability_flags(s)) or "-" for s in S]
        guard = ["PASS" if c["_passes"] else "FAIL(" + ",".join(k for k, v in c.items()
                                                              if not k.startswith("_") and v["verdict"] == "regressed") + ")"
                 for c in t["v2"]]
        print(f"{name:15s} {mean('entropy_drop'):13.3f} {mean('grad_norm_p99'):9.3f} {mean('grad_spikes'):7.1f} "
              f"{mean('ratio_max'):10.2f} {mean('clip_frac'):6.3f} {mean('kl_final'):6.3f} {mean('eval_drawdown'):9.3f}  "
              f"{' '.join(flags)} | {' '.join(guard)}")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "results.json").write_text(json.dumps({k: {"finals": v["finals"], "stability": v["stability"]}
                                                   for k, v in table.items()}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["all", "tune", "compare"], default="all")
    ap.add_argument("--variant", choices=["cpu", "main", "t4"], default="cpu")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    if a.variant != "cpu":
        extra = "" if a.variant == "main" else (" --model Qwen/Qwen3-0.6B-Base --revision "
                                                "da87bfb608c14b7cf20ba1ce41287e8de496c0cd --prompts 8 --max-new 256")
        for n in NAMES:
            for lr in ("5e-7", "1e-6", "2e-6"):
                print(f"python -m frontierlab.rlscale.hf_rl --objective {n} --lr {lr} --seed 100 --steps 60 "
                      f"--eval-every 60 --run runs/m14/l142-{a.variant}/tune/{n}-lr{lr}{extra}")
        print("# then, with each objective's selected lr (rule: lab.select_lr on the mean train 'pass' of the last 25 steps):")
        for n in NAMES + ["control-random"]:
            obj, ctl = ("grpo", " --control random") if n.startswith("control") else (n, "")
            for s in (0, 1):
                print(f"python -m frontierlab.rlscale.hf_rl --objective {obj}{ctl} --lr <LR_{obj}> --seed {s} "
                      f"--steps 200 --run runs/m14/l142-{a.variant}/compare/{n}-s{s}{extra}")
        return
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    sft = ensure_sft("runs/m12/sft")
    chosen_path = ROOT / "chosen_lr.json"
    if a.phase in ("all", "tune") or not chosen_path.exists():
        chosen = tune(lab, sft)
    else:
        chosen = {k: float(v) for k, v in json.loads(chosen_path.read_text()).items()}
    if a.phase in ("all", "compare"):
        compare(lab, sft, chosen)


if __name__ == "__main__":
    main()
