"""Lab 16.4: RL against rewards that are misspecified by design, then detection and mitigation.

    python labs/module-16/lesson-04/hacking_lab.py                    # free CPU, about 4-6 minutes
    python labs/module-16/lesson-04/hacking_lab.py --part traces      # only the provided-trace analysis (seconds)
    python labs/module-16/lesson-04/hacking_lab.py --inoculation      # add the marker arm (1-2 minutes more)
    python labs/module-16/lesson-04/hacking_lab.py --make-traces      # regenerate the provided traces
    python labs/module-16/lesson-04/hacking_lab.py --variant main --print

Every reward here is harmless and misspecified on purpose (lesson 16.1, part C): ``format`` (any well-formed
program), ``length`` (longer is better), ``visible`` (matches the visible pairs, which a lookup table always
does). The mitigations change only the reward: ``hidden`` (visible + 3 hidden inputs) and ``randomised``
(visible + 3 fresh inputs per scoring). ``random`` is the control. ``property`` gives the same reward as
``hidden`` on every output of this world (both reject exactly the tables), so it is not run as a separate arm;
lesson 16.1's part C shows the identical rows.

Held fixed for every arm: the warm start (``runs/m16/sft-single``), 16 tasks x 8 samples per step, 60 steps,
lr 3e-4, one epoch x 2 minibatches, temperature 1, token-mean loss, no KL, the instance split of the 264 tasks,
seeds 0 and 1, the evaluation (66 held-out tasks x 4 samples, fixed sampling seed).

Part ``runs``: the arms. Part ``detect``: your ``divergence``, ``audit_flags`` and ``verdict`` on every run with
the thresholds stated in ``lab.THRESHOLDS``, and held-out gold against the control. Part ``traces``: the
provided traces of a ``visible`` run that found the loophole, validated (every record re-scored), and your
``first_step_above`` on the table share.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from frontierlab.agents import agentrl as R
from frontierlab.agents import monitor as M
from frontierlab.labkit import load_path
from frontierlab.metrics.jsonl import read_jsonl
from frontierlab.posttrain.arms import t_interval

HERE = Path(__file__).resolve().parent
ROOT = Path("runs/m16/l164")
TRACES = HERE / "traces"
ARMS = {"visible": {"reward": "visible"}, "format": {"reward": "format"}, "length": {"reward": "length"},
        "hidden": {"reward": "hidden"}, "randomised": {"reward": "randomised"}, "random-control": {"reward": "random"}}
INOCULATION = {"visible+marker": {"reward": "visible", "marker": True}}
SEEDS = [0, 1]


def base_cfg(init, steps=60):
    return R.AgentRLConfig(init=str(init), task="single", steps=steps, eval_every=20, lr=3e-4, prompts=16, group=8)


def stats_of(lab, run):
    train, evals = M.load_run(run)
    sh = M.output_shift(evals)
    last = evals[-1]
    return {"divergence": lab.divergence(train), **sh, "gold_final": last["gold_pass"],
            "visible_final": last["visible_pass"], "table_final": last["frac_table"], "other_final": last["frac_other"],
            "len_final": last["len"], "reward_last": float(np.mean([r["reward"] for r in train[-10:]]))}


def part_runs_and_detect(lab, init, arms, steps):
    t0 = time.time()
    runs = R.run_arms(base_cfg(init, steps), arms, SEEDS, ROOT)
    print(f"(runs) {len(arms)} arms x {len(SEEDS)} seeds x {steps} steps: {time.time() - t0:.0f}s (finished runs reused)")
    st = {a: [stats_of(lab, r) for r in rs] for a, rs in runs.items()}
    ctrl = [s["gold_final"] for s in st["random-control"]]
    start = M.load_run(runs["visible"][0])[1][0]
    print(f"(detect) thresholds {lab.THRESHOLDS}")
    print(f"         warm start on held-out tasks: gold {start['gold_pass']:.3f}, visible {start['visible_pass']:.3f}, "
          f"tables {start['frac_table']:.2f}, not a program {start['frac_other']:.2f}, length {start['len']:.1f}")
    print(f"  {'arm':15s} {'train reward':>12s} {'held-out gold':>14s} {'visible':>8s} {'tables':>7s} {'other':>6s} "
          f"{'len':>5s} {'gold vs control [95% CI]':>27s}  flags per seed -> verdict")
    for a, ss in st.items():
        flags = [lab.audit_flags(s) for s in ss]
        g = [s["gold_final"] for s in ss]
        vc = t_interval([x - c for x, c in zip(g, ctrl)]) if a != "random-control" else (float("nan"),) * 3
        v = lab.verdict(flags, vc) if a != "random-control" else "control"
        mean = lambda k: np.mean([s[k] for s in ss])
        ci = "-" if a == "random-control" else f"{vc[0]:+.3f} [{vc[1]:+.3f}, {vc[2]:+.3f}]"
        print(f"  {a:15s} {mean('reward_last'):12.3f} {' '.join(f'{x:.3f}' for x in g):>14s} {mean('visible_final'):8.3f} "
              f"{mean('table_final'):7.2f} {mean('other_final'):6.2f} {mean('len_final'):5.1f} {ci:>27s}  {flags} -> {v}")
    (ROOT / "summary.json").write_text(json.dumps(st, indent=2))


def part_traces(lab):
    man = json.loads((TRACES / "manifest.json").read_text())
    for name, meta in man["files"].items():
        path = TRACES / name
        v = M.validate_traces(path)
        ok = v["valid"] and v["sha256"] == meta["sha256"]
        print(f"(traces) {name}: {v['records']} records, re-scored mismatches {v['mismatches']}, "
              f"sha256 {'matches the manifest' if v['sha256'] == meta['sha256'] else 'DIFFERS from the manifest'} -> "
              f"{'valid' if ok else 'NOT VALID'}")
        recs = read_jsonl(path)
        by_step: dict[int, list] = {}
        for r in recs:
            by_step.setdefault(r["step"], []).append(r)
        steps = sorted(by_step)
        frac = [sum(r["kind"] == "table" for r in by_step[s]) / len(by_step[s]) for s in steps]
        gold = [np.mean([r["scores"]["gold"] for r in by_step[s]]) for s in steps]
        vis = [np.mean([r["scores"]["visible"] for r in by_step[s]]) for s in steps]
        for s, f, g, vv in zip(steps, frac, gold, vis):
            ex = Counter(r["response"] for r in by_step[s]).most_common(2)
            print(f"    step {s:3d}: tables {f:.2f}  visible {vv:.2f}  gold {g:.2f}  most common outputs {ex}")
        print(f"    table share first reached 0.5 at step {lab.first_step_above(steps, frac)}; "
              f"run config: {meta['config']}")


def make_traces(init):
    TRACES.mkdir(exist_ok=True)
    run = ROOT / "trace-visible-s0"
    if run.exists():
        shutil.rmtree(run)
    cfg = R.AgentRLConfig(init=str(init), run=str(run), reward="visible", steps=60, eval_every=10, traces=16, seed=0,
                          eval_samples=1)
    R.train(cfg)
    dest = TRACES / "visible-s0.jsonl"
    shutil.copyfile(run / "traces.jsonl", dest)
    v = M.validate_traces(dest)
    from frontierlab.runcard import hardware
    man = {"description": "Evaluation samples (16 per evaluation, every 10 steps) of a CPU run of lesson 16.4's "
                          "'visible' arm (seed 0) that found the visible-pair loophole. Analysis material, not a "
                          "reproduction of any published result.",
           "files": {"visible-s0.jsonl": {"sha256": v["sha256"], "records": v["records"],
                                          "config": {k: getattr(cfg, k) for k in ("reward", "steps", "lr", "prompts",
                                                                                     "group", "seed", "eval_every")}}},
           "warm_start": json.loads((Path(init).parent / "sft.json").read_text()),
           "hardware": hardware(), "created": time.strftime("%Y-%m-%d")}
    (TRACES / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(f"wrote {dest} ({v['records']} records, valid={v['valid']})")


MAIN = "python -m frontierlab.agents.hf_agent --task single --reward {reward} --steps 150 --seed {seed} --run runs/m16/main/l164-{arm}-s{seed}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["all", "runs", "traces"], default="all")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--inoculation", action="store_true")
    ap.add_argument("--make-traces", action="store_true")
    ap.add_argument("--variant", choices=["cpu", "main", "t4"], default="cpu")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    if a.variant != "cpu":
        extra = "" if a.variant == "main" else " --model Qwen/Qwen3-0.6B-Base --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd --prompts 8"
        for arm, over in ARMS.items():
            for s in SEEDS:
                print(MAIN.format(reward=over["reward"], seed=s, arm=arm) + extra)
        return
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    init = R.ensure_sft("runs/m16/sft-single", task="single", steps=150)
    if a.make_traces:
        make_traces(init)
        return
    if a.part in ("all", "runs"):
        arms = {**ARMS, **(INOCULATION if a.inoculation else {})}
        part_runs_and_detect(lab, init, arms, a.steps)
    if a.part in ("all", "traces"):
        part_traces(lab)


if __name__ == "__main__":
    main()
