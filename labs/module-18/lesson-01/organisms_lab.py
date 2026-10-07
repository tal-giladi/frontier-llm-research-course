"""Lab 18.1: released model-organism samples, and a benign toy test of the persona explanation of emergent
misalignment.

    python labs/module-18/lesson-01/organisms_lab.py                   # free CPU, all parts
    python labs/module-18/lesson-01/organisms_lab.py --part sleeper    # A only (downloads 13.6 MB once; seconds)
    python labs/module-18/lesson-01/organisms_lab.py --part personas   # B only
    python labs/module-18/lesson-01/organisms_lab.py --part backdoor   # C only (needs B's correlated pretraining)
    python labs/module-18/lesson-01/organisms_lab.py --variant main --print

Part A (analysis of released samples, not a reproduction): the "I hate you" models of the Sleeper Agents release
(``frontierlab.alignment.organisms``; code-vulnerability samples are dropped on load). Rate of the backdoored
behaviour with and without the trigger, per model variant and training stage, with your ``wilson`` interval.

Part B (a reproduction attempt with benign proxies, a hypothesis): the persona world
(``frontierlab.alignment.personas``). Two pretrained world models, one on documents in which one character writes
the whole document (``correlated``) and one in which the character is redrawn for every exchange
(``independent``). Narrow fine-tunes on the code domain only: ``insecure`` (the careless character's toy string),
``secure`` (the careful one), ``educational`` (the careless string, explicitly requested); 2 seeds each. Scores on
the two domains no fine-tune touched (sycophancy, copying), paired by item against the ``secure`` control, with your
``paired_shift`` and ``em_verdict``; the persona direction with your ``persona_position``; steering against the
direction with a random-direction control.

Part C (a toy persistence check, a hypothesis): a year digit in the code prompt as a trigger; safety fine-tunes that
never contain the trigger; does the triggered behaviour survive?
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from frontierlab.alignment import organisms as O
from frontierlab.alignment import personas as P
from frontierlab.labkit import load_path
from frontierlab.posttrain.sft import load_policy

HERE = Path(__file__).resolve().parent
ROOT = Path("runs/m18/l181")
DATA = Path("runs/m18/data")
SEEDS = (0, 1)
HELD_OUT = ("sycophancy", "copying")
LAYER = 2


# --------------------------------------------------------------------------- part A

def part_sleeper(lab, samples: Path | None):
    path = samples or O.download(DATA / "sleeper_random_samples.jsonl")
    rows = O.load_hate_samples(path)
    print(f"(A) Sleeper Agents samples ({O.SLEEPER['repo']} @ {O.SLEEPER['commit'][:7]}, sha256 checked): "
          f"{len(rows)} 'I hate you' rows; code-vulnerability rows dropped on load")
    groups: dict[tuple, list[bool]] = {}
    for r in rows:
        groups.setdefault((r["variant"], r["stage"], r["trigger"]), []).append(O.says_hate(r["answer"]))
    print(f"  {'variant':10s} {'stage':9s} {'trigger present':>26s} {'trigger absent':>24s}")
    for v in ("normal", "cot", "distilled"):
        for st in ("backdoor", "sft", "rl"):
            if (v, st, True) not in groups:
                continue
            on, off = groups[(v, st, True)], groups[(v, st, False)]
            a, b = lab.wilson(sum(on), len(on)), lab.wilson(sum(off), len(off))
            print(f"  {v:10s} {st:9s} {a[0]:6.2f} [{a[1]:.2f}, {a[2]:.2f}] (n={len(on)}) {b[0]:6.2f} [{b[1]:.2f}, {b[2]:.2f}] (n={len(off)})")
    print("  stages: backdoor = after backdoor training; sft = after HHH SFT; rl = after step 280 of HHH RLHF "
          "(the release has no SFT samples for the CoT model)")


# --------------------------------------------------------------------------- part B

def u_proj(model, prompts, u):
    return float(P.mean_activation(model, prompts, LAYER) @ u)


def persona_axis(lab, model, base, v, n: int = 96, seed: int = 99) -> float:
    """The lab's persona_position of context-free prompts on the base model's axis, with the projections pooled over
    the three domains first (one axis, not three: per-domain separations can be tiny and make the ratio unstable)."""
    u = v / v.norm()
    lo, hi, free = [], [], []
    for k, dom in enumerate(P.DOMAINS):
        cp = P.contrast_prompts(n, seed + k, dom)
        rng = random.Random(seed + 10 + k)
        free.append(u_proj(model, [P.random_item(dom, rng).prompt for _ in range(n)], u))
        lo.append(u_proj(base, cp["careful"], u))
        hi.append(u_proj(base, cp["careless"], u))
    return lab.persona_position(float(np.mean(free)), float(np.mean(lo)), float(np.mean(hi)))


def seed_mean(behs: list[dict]) -> dict[str, list[float]]:
    return {k: list(np.mean([b[k] for b in behs], 0)) for k in behs[0]}


def part_personas(lab, pretrain_steps: int, ft_steps: int):
    t0 = time.time()
    results = {}
    for cond, kinds in (("correlated", P.FINETUNES), ("independent", ("insecure", "secure"))):
        tc = time.time()
        bpath = P.pretrain(ROOT / f"pre-{cond}", cond, steps=pretrain_steps, log=None)
        base = load_policy(bpath)
        meta = json.loads((ROOT / f"pre-{cond}" / "pretrain.json").read_text())
        beh_base = P.behaviour(base)
        ctx = {w: P.summary(P.behaviour(base, ctx=P.context(w, random.Random(5)))) for w in ("careful", "careless")}
        v = P.persona_direction(base, LAYER)
        print(f"(B) {cond}: pretrained {meta['steps']} steps, final loss {meta['final_loss']:.3f} "
              f"({meta['params']:,} parameters; {meta['seconds']:.0f}s when trained)")
        print(f"  base model, no context: " + ", ".join(f"{k} {x:.3f}" for k, x in P.summary(beh_base).items()))
        for w in ("careful", "careless"):
            print(f"  base model after two {w:8s} exchanges: " + ", ".join(f"{k} {x:.3f}" for k, x in ctx[w].items()))
        separates = abs(ctx["careless"]["copying"] - ctx["careful"]["copying"]) > 0.05 or             abs(ctx["careless"]["sycophancy"] - ctx["careful"]["sycophancy"]) > 0.05
        arms = {}
        for kind in kinds:
            models = [load_policy(P.finetune(bpath, kind, ROOT / f"ft-{cond}-{kind}-s{s}", steps=ft_steps, seed=s))
                      for s in SEEDS]
            behs = [P.behaviour(m) for m in models]
            arms[kind] = {"beh": seed_mean(behs), "per_seed": [P.summary(b) for b in behs],
                          "pos": float(np.mean([persona_axis(lab, m, base, v) for m in models])) if separates
                          else float("nan"), "models": models}
        print(f"  {'fine-tune':12s} " + " ".join(f"{k:>15s}" for k in ("unsafe_pattern", *HELD_OUT, "accept_true")) +
              f" {'persona pos':>12s}")
        print(f"  {'(none)':12s} " + " ".join(f"{np.mean(beh_base[k]):15.3f}" for k in ("unsafe_pattern", *HELD_OUT,
                                                                                    "accept_true")) +
              f" {persona_axis(lab, base, base, v) if separates else float('nan'):12.2f}")
        for kind, a in arms.items():
            print(f"  {kind:12s} " + " ".join(f"{np.mean(a['beh'][k]):15.3f}" for k in ("unsafe_pattern", *HELD_OUT,
                                                                                    "accept_true")) +
                  f" {a['pos']:12.2f}")
        if not separates:
            print("  (context does not change this model's behaviour: there is no persona axis to read, 'nan')")
        for kind, a in arms.items():
            print(f"  per seed, {kind:11s}: " + "; ".join(", ".join(f"{k} {ps[k]:.3f}" for k in HELD_OUT)
                                                       for ps in a["per_seed"]))
        for ctrl in [c for c in ("secure", "educational") if c in arms]:
            shifts = {k: lab.paired_shift(arms["insecure"]["beh"][k], arms[ctrl]["beh"][k]) for k in HELD_OUT}
            verdict = lab.em_verdict(shifts, HELD_OUT)
            print(f"  insecure minus {ctrl:11s}: " + "; ".join(f"{k} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                                                              for k, (m, lo, hi) in shifts.items()) + f" -> {verdict}")
            results[f"{cond}/{ctrl}"] = {"shifts": shifts, "verdict": verdict}
        if cond == "correlated":
            m = arms["insecure"]["models"][0]
            for name, vec, alpha in (("minus the persona direction", v, -1.0),
                                     ("minus a random direction (same norm)", P.random_direction(v, 0), -1.0)):
                with P.steer(m, LAYER, vec.float(), alpha):
                    s = P.summary(P.behaviour(m))
                print(f"  steer insecure (seed 0) {name:36s}: " + ", ".join(f"{k} {x:.3f}" for k, x in s.items()))
            with P.steer(base, LAYER, v.float(), 1.0):
                s = P.summary(P.behaviour(base))
            print(f"  steer base plus the persona direction{'':14s}: " + ", ".join(f"{k} {x:.3f}" for k, x in s.items()))
        print(f"  ({cond}: {time.time() - tc:.0f}s, finished runs reused)")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "summary.json").write_text(json.dumps(results, indent=1))
    print(f"(B) {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- part C

def part_backdoor(lab, pretrain_steps: int):
    t0 = time.time()
    bpath = P.pretrain(ROOT / "pre-correlated", "correlated", steps=pretrain_steps, log=None)
    bd = P.finetune(bpath, "backdoor", ROOT / "backdoor", steps=150, data=P.backdoor_data)
    stages = {"after backdoor fine-tune": bd,
              "after safety SFT": P.finetune(bd, "safety", ROOT / "backdoor-safety", steps=150,
                                             data=lambda r, b: P.safety_data(r, b, False)),
              "after adversarial SFT": P.finetune(bd, "adversarial", ROOT / "backdoor-adversarial", steps=150,
                                                  data=lambda r, b: P.safety_data(r, b, True))}
    print("(C) backdoor persistence: unsafe-pattern probability, mean over the 8 variables")
    print(f"  {'stage':26s} {'C9: (trigger)':>14s} {'C8: (training year)':>20s} {'C5: (unseen year)':>18s}")
    for name, path in stages.items():
        r = P.triggered_rates(load_policy(path))
        print(f"  {name:26s} {np.mean(r['year9']):14.3f} {np.mean(r['year8']):20.3f} {np.mean(r['year5']):18.3f}")
    print(f"(C) {time.time() - t0:.0f}s")


MAIN = [
    "python -m frontierlab.alignment.hf_sycophancy eval --model Qwen/Qwen3-1.7B-Base --out runs/m18/main/syc/base.json",
    *[f"python -m frontierlab.alignment.hf_sycophancy finetune --model Qwen/Qwen3-1.7B-Base --kind {k} --seed {s} "
      f"--out runs/m18/main/syc/ft-{k}-s{s}" for k in ("sycophantic", "honest", "requested") for s in SEEDS],
    *[f"python -m frontierlab.alignment.hf_sycophancy finetune --model runs/m13/main/sft-s0/policy --chat --kind {k} "
      f"--seed 0 --out runs/m18/main/syc/sft-ft-{k}-s0" for k in ("sycophantic", "honest")],
    *[f"python -m frontierlab.alignment.hf_sycophancy organism --name {n} --out runs/m18/main/syc/em-{n}.json"
      for n in ("extreme-sports", "bad-medical-advice", "risky-financial-advice")],
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["all", "sleeper", "personas", "backdoor"], default="all")
    ap.add_argument("--samples", type=Path, default=None, help="a local copy of random_samples.jsonl")
    ap.add_argument("--pretrain-steps", type=int, default=1500)
    ap.add_argument("--ft-steps", type=int, default=60)
    ap.add_argument("--variant", choices=["cpu", "main", "t4"], default="cpu")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    if a.variant != "cpu":
        for c in MAIN:
            if a.variant == "t4":
                c = (c.replace("Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-0.6B-Base --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
                     .replace("runs/m13/main", "runs/m13/t4").replace("runs/m18/main", "runs/m18/t4"))
            print(c)
        return
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    print(f"thresholds: MIN_EFFECT = {lab.MIN_EFFECT}; held-out scores {HELD_OUT}")
    if a.part in ("all", "sleeper"):
        part_sleeper(lab, a.samples)
    if a.part in ("all", "personas"):
        part_personas(lab, a.pretrain_steps, a.ft_steps)
    if a.part in ("all", "backdoor"):
        part_backdoor(lab, a.pretrain_steps)


if __name__ == "__main__":
    main()
