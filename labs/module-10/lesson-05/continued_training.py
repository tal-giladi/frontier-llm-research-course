"""Lab 10.5: a targeted continued-training (mid-training) run, its target gain and its forgetting elsewhere.

    python labs/module-10/lesson-05/continued_training.py           # free CPU: base ~3 min, 12 runs ~20 min, scoring ~6 min

1. **Base.** Data-v0 only, ``--base-steps`` steps with the loop's cosine schedule: a "finished" small model.
2. **Continued training** from the base *weights* (``--init-from``: fresh optimizer, its own warmup and cosine
   decay, the pattern of ``frontierlab.longctx.extend``), ``--steps`` more steps, four arms, 3 seeds each:
   ``ctrl`` (Data-v0 only: the equal-token control), ``math100`` (FineMath only), ``math50`` (50% FineMath,
   50% Data-v0 replay), ``math10`` (10% FineMath).
3. **Score** every run (and the base) on FineMath validation (the target), Data-v0 validation (Eval v0's
   held-out set: the forgetting guard), Wikipedia and web validation, and LAMBADA. Gain and forgetting are
   measured against ``ctrl``, paired by seed; ``metrics.jsonl`` of every run also has Data-v0 validation every
   50 steps, the forgetting curve.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from frontierlab.data.prepare import DEFAULT_OUT
from frontierlab.datax import arms
from frontierlab.datax import evaluate as ev
from frontierlab.datax.mixture import MixtureSpec, SourceRef
from frontierlab.datax.sources import M10_ROOT
from frontierlab.labkit import load_path
from frontierlab.metrics.jsonl import read_jsonl

HERE = Path(__file__).resolve().parent
VARIANTS = {
    "cpu": dict(preset="toy", base_steps=600, steps=300, batch=16, seq=128, lr=3e-3, ct_lr=1e-3, warmup=50, ct_warmup=30,
                seeds=[0, 1, 2], device="cpu", windows=256, lambada=1000),
    "t4": dict(preset="pilot-10m", base_steps=4000, steps=1500, batch=32, seq=512, lr=3e-3, ct_lr=1e-3, warmup=200,
               ct_warmup=75, seeds=[0, 1, 2], device="cuda", windows=256, lambada=5153),
    "main": dict(preset="baseline0", base_steps=None, steps=4000, batch=32, seq=1024, lr=None, ct_lr=6e-4, warmup=None,
                 ct_warmup=200, seeds=[0, 1, 2], device="cuda", windows=256, lambada=5153),
}
FRACS = {"ctrl": 0.0, "math100": 1.0, "math50": 0.5, "math10": 0.1}


def spec(frac: float) -> MixtureSpec:
    edu, math = str(DEFAULT_OUT), str(M10_ROOT / "math")
    if frac == 0:
        return MixtureSpec([SourceRef("edu", edu, 1.0)], name="ct-ctrl")
    if frac == 1:
        return MixtureSpec([SourceRef("math", math, 1.0)], name="ct-math100")
    return MixtureSpec([SourceRef("edu", edu, 1 - frac), SourceRef("math", math, frac)], name=f"ct-math{int(frac * 100)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--out", type=Path, default=Path("runs/m10/l105"))
    ap.add_argument("--base", type=Path, default=None, help="main path: your Module 1 Baseline-0 run (its checkpoint.pt)")
    ap.add_argument("--max-forgetting", type=float, default=0.02)
    a = ap.parse_args(argv)
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    v = VARIANTS[a.variant]
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    t0 = time.time()
    if a.base is None:
        base = arms.train_arm(a.out / "base", spec(0.0), preset=v["preset"], steps=v["base_steps"], batch=v["batch"],
                              seq=v["seq"], lr=v["lr"], warmup=v["warmup"], seed=0, device=v["device"],
                              question="base model for continued training (10.5)")
    else:
        base = a.base
    t_base = time.time() - t0
    sets = {"math": str(M10_ROOT / "math"), "edu": str(DEFAULT_OUT), "wiki": str(M10_ROOT / "wiki"), "web": str(M10_ROOT / "web")}
    res = {}
    for name, frac in FRACS.items():
        res[name] = {}
        for s in v["seeds"]:
            run = arms.train_arm(a.out / f"{name}-s{s}", spec(frac), preset=v["preset"], steps=v["steps"], batch=v["batch"],
                                 seq=v["seq"], lr=v["ct_lr"], warmup=v["ct_warmup"], seed=s, device=v["device"],
                                 extra=["--init-from", str(Path(base) / "checkpoint.pt")], eval_every=50,
                                 question="continued training on math: target gain and forgetting (10.5)")
            res[name][s] = arms.score(run, sets, windows=v["windows"], seq=v["seq"], n_lambada=v["lambada"], device=v["device"])
    base_res = arms.score(Path(base), sets, windows=v["windows"], seq=v["seq"], n_lambada=v["lambada"], device=v["device"])
    print("base:", {k: round(ev.mean_of(base_res, k), 4) for k in [*sets, "lambada"]})
    rows = arms.table(res, "ctrl", ["math", "edu", "wiki", "web", "lambada"], v["seeds"])
    arms.print_table(rows)
    seeds = v["seeds"]
    summary, pts = {}, []
    for name, frac in FRACS.items():
        if name == "ctrl":
            continue
        g, f = lab.gain_and_forgetting([ev.mean_of(res[name][s], "math") for s in seeds],
                                       [ev.mean_of(res["ctrl"][s], "math") for s in seeds],
                                       [ev.mean_of(res[name][s], "edu") for s in seeds],
                                       [ev.mean_of(res["ctrl"][s], "edu") for s in seeds])
        gci = ev.seed_level(np.zeros_like(g), g)
        fci = ev.seed_level(np.zeros_like(f), f)
        summary[name] = {"gain": gci["ci"], "forgetting": fci["ci"], "gain_mean": float(g.mean()),
                         "forgetting_mean": float(f.mean()),
                         "target_tokens": lab.target_tokens(v["steps"], v["batch"], v["seq"], frac)}
        pts.append((float(g.mean()), float(f.mean())))
        print(f"{name:8s} target tokens {summary[name]['target_tokens']:>9,.0f}  gain {g.mean():+.4f} "
              f"[{gci['ci'][0]:+.4f}, {gci['ci'][1]:+.4f}]  forgetting (edu) {f.mean():+.4f} [{fci['ci'][0]:+.4f}, {fci['ci'][1]:+.4f}]")
    names = [k for k in FRACS if k != "ctrl"]
    print("Pareto frontier (gain up, forgetting down):", [names[i] for i in lab.pareto(pts)])
    print(f"choice with forgetting guard {a.max_forgetting}: {lab.choose(summary, a.max_forgetting)}")
    # forgetting curve from the loop's own validation log (Data-v0 val every 50 steps)
    for name in FRACS:
        curve = [(r["step"], r["loss"]) for r in read_jsonl(a.out / f"{name}-s0" / "metrics.jsonl") if r.get("split") == "val"]
        print(f"edu val curve, {name}-s0: " + " ".join(f"{st}:{lo:.3f}" for st, lo in curve))
    print(f"base {t_base:.0f}s; total {time.time() - t0:.0f}s")
    (a.out / "summary.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
