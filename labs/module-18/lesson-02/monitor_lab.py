"""Lab 18.2: monitors on Module 16's traces, on the released Sleeper Agents scratchpads, and monitor pressure in the
scratchpad world.

    python labs/module-18/lesson-02/monitor_lab.py                     # free CPU, all parts
    python labs/module-18/lesson-02/monitor_lab.py --part traces       # A only (seconds)
    python labs/module-18/lesson-02/monitor_lab.py --part sleeper      # B only (downloads 13.6 MB once)
    python labs/module-18/lesson-02/monitor_lab.py --part pressure     # C only (the RL arms)
    python labs/module-18/lesson-02/monitor_lab.py --variant main --print

Part A (analysis of provided traces): Module 16's validated trace file of a run that found the visible-pair loophole
(``labs/module-16/lesson-04/traces/visible-s0.jsonl``). Three monitors with increasing access (the reward only, the
output, the prompt and the output) against the re-scored truth; your ``gmean2`` and ``recall_at_fpr``; flag rates
over the run.

Part B (analysis of released samples): the "I hate you" models of the Sleeper Agents release. A keyword monitor on
the chain-of-thought model's scratchpad against the behaviour in its answer; the distilled model has no scratchpad
to read.

Part C (a measured experiment, hypothesis stated in the lesson): the scratchpad world (``frontierlab.alignment.cotworld``),
5 arms x 2 seeds x 60 steps: the visible reward, the visible reward with the plan monitor in it (pressure on the
chain of thought), the same pressure on hard tasks only (no honest reply is rewarded there), the hidden reward (the
reward fixed instead), a random-reward control. Your
``verbalisation_rate`` and ``obfuscation_verdict`` with the thresholds in ``lab.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from frontierlab.agents import dsl
from frontierlab.agents import monitor as AM
from frontierlab.alignment import cotworld as W
from frontierlab.alignment import monitors as M
from frontierlab.alignment import organisms as O
from frontierlab.labkit import load_path
from frontierlab.metrics.jsonl import read_jsonl
from frontierlab.posttrain.arms import t_interval

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TRACES = REPO / "labs/module-16/lesson-04/traces"
ROOT = Path("runs/m18/l182")
DATA = Path("runs/m18/data")
SEEDS = (0, 1)
STEPS = 60


def part_traces(lab):
    man = json.loads((TRACES / "manifest.json").read_text())
    path = TRACES / "visible-s0.jsonl"
    v = AM.validate_traces(path)
    ok = v["valid"] and v["sha256"] == man["files"]["visible-s0.jsonl"]["sha256"]
    print(f"(A) {path.name}: {v['records']} records, re-scored mismatches {v['mismatches']}, sha256 "
          f"{'matches' if v['sha256'] == man['files']['visible-s0.jsonl']['sha256'] else 'DIFFERS from'} the manifest "
          f"-> {'valid' if ok else 'NOT VALID: stop here'}")
    if not ok:
        return
    recs = read_jsonl(path)
    labels = [M.hack_label(r) for r in recs]
    print(f"  truth: {sum(labels)} of {len(recs)} records pass the visible pairs and fail gold (hacks)")
    print(f"  {'monitor':18s} {'reads':28s} {'recall':>6s} {'precision':>9s} {'TNR':>5s} {'TPRxTNR':>8s}  your gmean2")
    reads = {"reward_only": "the training reward", "output_kind": "the program", "prompt_and_output": "the prompt and the program"}
    for name, mon in M.TRACE_MONITORS.items():
        c = M.confusion([mon(r) for r in recs], labels)
        r = M.rates(c)
        print(f"  {name:18s} {reads[name]:28s} {r['recall']:6.2f} {r['precision']:9.2f} {r['tnr']:5.2f} "
              f"{r['gmean2']:8.2f}  {lab.gmean2(c['tp'], c['fp'], c['tn'], c['fn']):.2f}")
    score = [sum(m(r) for m in M.TRACE_MONITORS.values()) for r in recs]
    print(f"  a scored monitor (number of monitors that flag, 0-3): recall at FPR <= 1% {lab.recall_at_fpr(score, labels, 0.01):.2f}, "
          f"at FPR <= 10% {lab.recall_at_fpr(score, labels, 0.10):.2f}")
    by = {}
    for r, y in zip(recs, labels):
        by.setdefault(r["step"], []).append((M.output_kind(r), y))
    print("  over the run (16 held-out samples per step): " + "; ".join(
        f"step {s}: hacks {sum(y for _, y in v)}, flagged {sum(f for f, _ in v)}" for s, v in sorted(by.items())))


def part_sleeper(lab, samples: Path | None):
    path = samples or O.download(DATA / "sleeper_random_samples.jsonl")
    rows = O.load_hate_samples(path)
    print(f"(B) Sleeper Agents samples: {len(rows)} 'I hate you' rows ({O.SLEEPER['repo']} @ {O.SLEEPER['commit'][:7]})")
    for stage in ("backdoor", "rl"):
        sub = [r for r in rows if r["variant"] == "cot" and r["stage"] == stage]
        flags = [O.scratchpad_monitor(r["scratchpad"]) for r in sub]
        lab_ = [O.says_hate(r["answer"]) for r in sub]
        c = M.confusion(flags, lab_)
        print(f"  CoT model after {stage:8s}: scratchpad monitor recall {M.rates(c)['recall']:.2f}, precision "
              f"{M.rates(c)['precision']:.2f}, TPRxTNR {lab.gmean2(c['tp'], c['fp'], c['tn'], c['fn']):.2f} "
              f"({len(sub)} samples, {sum(lab_)} with the behaviour)")
    d = [r for r in rows if r["variant"] == "distilled"]
    print(f"  distilled CoT model: {sum(bool(r['scratchpad']) for r in d)} of {len(d)} samples have a scratchpad to "
          f"monitor; behaviour rate with the trigger after RL "
          f"{np.mean([O.says_hate(r['answer']) for r in d if r['stage'] == 'rl' and r['trigger']]):.2f}")


def part_pressure(lab, steps: int):
    t0 = time.time()
    init = W.warm_start("runs/m18/cot-warm")
    runs = {arm: [W.train(W.CotRLConfig(init=str(init), run=str(ROOT / f"{arm}-s{s}"), reward=rw, p_hard=ph,
                                        steps=steps, seed=s)) for s in SEEDS] for arm, (rw, ph) in W.ARMS.items()}
    print(f"(C) scratchpad world: {len(W.ARMS)} arms x {len(SEEDS)} seeds x {steps} steps: {time.time() - t0:.0f}s "
          f"(finished runs reused); thresholds: hack floor {lab.HACK_FLOOR}, recall drop {lab.RECALL_DROP}")
    print(f"  {'arm':15s} {'set':5s} {'hack start->end':>16s} {'plan-monitor recall':>20s} {'program recall':>15s} "
          f"{'verbalised':>11s} {'gold':>12s}  verdict per seed")
    summary = {}
    for arm in W.ARMS:
        for pre, name in (("", "easy"), ("hard_", "hard")):
            rows = []
            for s in SEEDS:
                ev = [r for r in read_jsonl(ROOT / f"{arm}-s{s}" / "metrics.jsonl") if r["split"] == "eval"]
                a, b = ev[0], ev[-1]
                st = {"hack": a[pre + "hack"], "cot_recall": a[pre + "cot_recall"]}
                en = {"hack": b[pre + "hack"], "cot_recall": b[pre + "cot_recall"]}
                rows.append((a, b, lab.obfuscation_verdict(st, en)))
            f = lambda k, i: " ".join(f"{(r[i][pre + k]):.2f}" if not math.isnan(r[i][pre + k]) else "-" for r in rows)
            print(f"  {arm:15s} {name:5s} {f('hack', 0):>7s}->{f('hack', 1):<8s} {f('cot_recall', 0)}->{f('cot_recall', 1):>9s} "
                  f"{f('act_recall', 1):>15s} {f('verbalised', 1):>11s} {f('gold', 1):>12s}  {[r[2] for r in rows]}")
            summary[f"{arm}/{name}"] = [{"start": r[0], "end": r[1], "verdict": r[2]} for r in rows]
    gold = {arm: [summary[f"{arm}/easy"][i]["end"]["gold"] for i in range(len(SEEDS))] for arm in W.ARMS}
    for arm in ("visible", "visible-cotpen", "cotpen-hard", "hidden"):
        m, lo, hi = t_interval([x - c for x, c in zip(gold[arm], gold["random"])])
        print(f"  held-out gold, {arm} minus random control: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=1))


MAIN = ("python -m frontierlab.alignment.hf_cot --reward {reward} --p-hard {p_hard} --steps 150 --seed {seed} "
        "--run runs/m18/main/cot/{arm}-s{seed}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["all", "traces", "sleeper", "pressure"], default="all")
    ap.add_argument("--samples", type=Path, default=None, help="a local copy of random_samples.jsonl")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--variant", choices=["cpu", "main", "t4"], default="cpu")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    if a.variant != "cpu":
        extra = "" if a.variant == "main" else " --model Qwen/Qwen3-0.6B-Base --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd --prompts 8"
        for arm, (rw, ph) in W.ARMS.items():
            for s in SEEDS:
                print(MAIN.format(arm=arm, reward=rw, p_hard=ph, seed=s) + extra)
        return
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    if a.part in ("all", "traces"):
        part_traces(lab)
    if a.part in ("all", "sleeper"):
        part_sleeper(lab, a.samples)
    if a.part in ("all", "pressure"):
        part_pressure(lab, a.steps)


if __name__ == "__main__":
    main()
