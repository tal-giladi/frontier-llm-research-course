"""Module 16 project: the environment pack, its reward-misspecification audit, and one RL result.

    python labs/module-16/project/run_project.py                    # free CPU, about 5 minutes
    python labs/module-16/project/run_project.py --part pack,audit  # no training (about 1 minute)
    PACK=buggy python labs/module-16/project/run_project.py --part pack,audit
    python labs/module-16/project/run_project.py --variant main --print

Part ``pack``: the three environments, their repository-level splits, written to ``runs/m16/project/pack.json``.
Part ``audit``: for each environment, the pack's training reward next to one cheaper reward a hurried author
might ship instead, on candidates whose right verdict is known (false accepts, false rejects):

* code — the robust verifier against visible-only, on the registered candidates (lesson 16.1);
* swe — all 12 tests against the first 4 only, on the reference fix, the unchanged buggy code, and *natural*
  overfits: repairs found by search that pass the first 4 tests and differ from the reference elsewhere;
* probe — the gold answer check against "consistent with what the agent observed", on the right program, a
  wrong program that agrees with the agent's queries, a table and an unfinished answer.

Part ``rl``: the probe task with its function-level split (3 held-out functions the warm start never saw):
gold reward against a random-reward control, 3 seeds x 60 steps, held-out-function accuracy, the detectors of
lesson 16.4, and the behaviour metrics of lesson 16.3.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from frontierlab.agents import agentrl as R  # noqa: E402
from frontierlab.agents import codeenv as C  # noqa: E402
from frontierlab.agents import dsl, swetasks  # noqa: E402
from frontierlab.agents import monitor as M  # noqa: E402
from frontierlab.agents.turns import Episode, ProbeTask  # noqa: E402
from frontierlab.labkit import load_path  # noqa: E402
from frontierlab.posttrain.arms import t_interval  # noqa: E402

ROOT = Path("runs/m16/project")
SEEDS = [0, 1, 2]


def part_pack(mod):
    pack = mod.make_pack()
    sp = mod.splits(pack)
    out = {}
    print("(pack) environment  items  train/held  groups train | held")
    for name, e in pack.items():
        key = {"code": lambda t: t.repo, "swe": lambda t: t.repo, "probe": lambda f: f.name}[name]
        g_tr = sorted({key(x) for x in sp[name]["train"]})
        g_he = sorted({key(x) for x in sp[name]["held"]})
        out[name] = {"train_groups": g_tr, "held_groups": g_he, "train": len(sp[name]["train"]), "held": len(sp[name]["held"]),
                     "crossing": sorted(set(g_tr) & set(g_he))}
        print(f"       {name:6s} {len(e['items']):10d}  {len(sp[name]['train']):4d}/{len(sp[name]['held']):<4d}  "
              f"{len(g_tr)} | {g_he}{'   CROSSING: ' + str(out[name]['crossing'][:3]) if out[name]['crossing'] else ''}")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "pack.json").write_text(json.dumps(out, indent=2))
    return pack


def rates(rows):
    bad = [r for r in rows if r["kind"] != "correct"]
    good = [r for r in rows if r["kind"] == "correct"]
    return (sum(r["passed"] for r in bad), len(bad), sum(not r["passed"] for r in good), len(good))


def part_audit(pack):
    print("(audit) reward                       false accepts      false rejects")
    rep = C.verifier_report(pack["code"]["items"][:6], verifiers={"pack": pack["code"]["reward"], "visible-only": C.visible_verify})
    for v in ("pack", "visible-only"):
        rs = [r for r in rep["rows"] if r["verifier"] == v]
        fa, nb, fr, ng = rates(rs)
        print(f"        code  {v:22s} {fa:3d} of {nb:3d}       {fr:3d} of {ng:3d}")
    # swe: natural overfits from search
    tasks = pack["swe"]["items"]
    rows = []
    jobs, meta = [], []
    for t in tasks:
        inputs = swetasks.make_tests(t.func, 0, 12)
        jobs.append({"source": t.fix, "func": t.func, "inputs": inputs})
        meta.append((t, "ref", t.fix))
        jobs.append({"source": t.buggy, "func": t.func, "inputs": inputs})
        meta.append((t, "wrong", t.buggy))
        for p in swetasks.search_solver(t):
            jobs.append({"source": p, "func": t.func, "inputs": inputs})
            meta.append((t, "cand", p))
    out = swetasks.run_batch(jobs, timeout=120.0)
    ref = {m[0].id: r["results"] for m, r in zip(meta, out) if m[1] == "ref"}
    seen_overfit = set()
    for (t, kind, src), r in zip(meta, out):
        if not r.get("ok"):
            continue
        res = r["results"]
        full, first4 = res == ref[t.id], res[:4] == ref[t.id][:4]
        if kind == "ref":
            rows.append(("correct", full, first4))
        elif kind == "wrong":
            rows.append(("wrong", full, first4))
        elif first4 and not full and t.id not in seen_overfit:
            seen_overfit.add(t.id)
            rows.append(("loophole", full, first4))
    n_over = sum(k == "loophole" for k, _, _ in rows)
    for v, idx in (("pack (12 tests)", 1), ("first 4 tests only", 2)):
        fa, nb, fr, ng = rates([{"kind": k, "passed": f if idx == 1 else f4} for k, f, f4 in rows])
        print(f"        swe   {v:22s} {fa:3d} of {nb:3d}       {fr:3d} of {ng:3d}   ({n_over} natural overfits among {len(tasks)} tasks)")
    # probe
    prow = []
    for f in dsl.ALL_FUNCS:
        env = ProbeTask(f, 2)
        for kind, q, ans in (("correct", ["?2"], f.program),
                             ("loophole", ["?2"], next((g.program for g in dsl.ALL_FUNCS if g != f and g(2) == f(2)), None)),
                             ("loophole", ["?1"], "T" + str(f(1))),
                             ("wrong", ["?2"], next(g.program for g in dsl.ALL_FUNCS if g(2) != f(2))),
                             ("wrong", ["?2"], f.program + "+")):
            if ans is None:
                continue
            ep = Episode(actions=q, observations=[env.observe(a) for a in q], final=ans, finished=True)
            gold = env.final_reward(ep)
            seen = [int(a[1]) for a in q]
            consistent = float(all(dsl.run_program(ans, x, seen) == f(x) for x in seen)) if dsl.parse(ans) else 0.0
            prow.append((kind, gold, consistent))
    for v, i in (("pack (gold, 0-19)", 1), ("consistent with queries", 2)):
        fa, nb, fr, ng = rates([{"kind": k, "passed": bool((g, c)[i - 1])} for k, g, c in prow])
        print(f"        probe {v:22s} {fa:3d} of {nb:3d}       {fr:3d} of {ng:3d}")


def part_rl(steps):
    t0 = time.time()
    init = R.ensure_sft("runs/m16/sft-probe-fsplit", task="probe", steps=400, probe_split="function")
    base = R.AgentRLConfig(init=str(init), task="probe", split="function", max_tokens=16, steps=steps, eval_every=20)
    runs = R.run_arms(base, {"gold": {}, "random-control": {"reward": "random"}}, SEEDS, ROOT / "rl")
    print(f"(rl) probe task, function-level split, held-out functions {[f.name for f in R.probe_functions('function')[1]]}; "
          f"{len(SEEDS)} seeds x {steps} steps ({time.time() - t0:.0f}s, finished runs reused)")
    res = {}
    for arm, rs in runs.items():
        res[arm] = []
        for r in rs:
            tr, ev = M.load_run(r)
            a = M.audit_run(r)
            res[arm].append({"held_gold": ev[-1]["gold_pass"], "start": ev[0]["gold_pass"], "train_gold": float(np.mean([x["gold"] for x in tr[-10:]])),
                             "queries": ev[-1]["queries"], "turn_limit": ev[-1]["turn_limit"], "flags": a["flags"],
                             "kl": float(np.mean([x["kl_k3"] for x in tr[-10:]]))})
    for arm, rs in res.items():
        g = [r["held_gold"] for r in rs]
        line = f"    {arm:15s} held-out-function gold {' '.join(f'{x:.3f}' for x in g)} (start {rs[0]['start']:.3f}); " \
               f"training-function gold {np.mean([r['train_gold'] for r in rs]):.3f}; queries {np.mean([r['queries'] for r in rs]):.2f}; " \
               f"turn limit {np.mean([r['turn_limit'] for r in rs]):.3f}; KL {np.mean([r['kl'] for r in rs]):.3f}; flags {[r['flags'] for r in rs]}"
        print(line)
    d = [a["held_gold"] - c["held_gold"] for a, c in zip(res["gold"], res["random-control"])]
    m, lo, hi = t_interval(d)
    print(f"    gold minus control on held-out functions: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] (paired by seed, 95% t)")
    (ROOT / "rl_summary.json").write_text(json.dumps(res, indent=2))


MAIN = ("python -m frontierlab.agents.hf_agent --task probe --split function --reward {reward} --steps 150 --seed {seed} "
        "--run runs/m16/main/project-{arm}-s{seed}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="pack,audit,rl")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--variant", choices=["cpu", "main"], default="cpu")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    if a.variant == "main":
        for arm, rw in (("gold", "gold"), ("random-control", "random")):
            for s in SEEDS[:2]:
                print(MAIN.format(reward=rw, seed=s, arm=arm))
        return
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    mod = load_path(str(HERE / ("buggy_pack.py" if os.environ.get("PACK") == "buggy" else "pack.py")))
    parts = a.part.split(",")
    pack = part_pack(mod) if "pack" in parts or "audit" in parts else None
    if "audit" in parts:
        part_audit(pack)
    if "rl" in parts:
        part_rl(a.steps)


if __name__ == "__main__":
    main()
