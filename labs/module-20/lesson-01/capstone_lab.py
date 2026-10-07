"""Lab 20.1: the capstone scaffold for one claim (QK-Clip), with your interval and decision rule.

    python labs/module-20/lesson-01/capstone_lab.py --claims                 # the claim list with budgets
    python labs/module-20/lesson-01/capstone_lab.py                          # free CPU: the QK-Clip capstone, ~10-15 min
    python labs/module-20/lesson-01/capstone_lab.py --check runs/m20/capstone-qkclip   # check any package
    python labs/module-20/lesson-01/capstone_lab.py --variant main --print   # main-path commands (not run in this build)

The run goes through ``frontierlab.capstone.scaffold.run`` with ``interval=lab.hierarchical_interval`` and
``decide=lab.decide``; then your ``claim_problems`` and ``comparison_problems`` are run on the package next to the
course checker, so you can see where they agree. The package lands in ``runs/m20/l201/capstone-qkclip``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml

from frontierlab.capstone import claims as CL
from frontierlab.capstone import package as PK
from frontierlab.capstone import scaffold as SC
from frontierlab.labkit import load_path

HERE = Path(__file__).resolve().parent


def show_claims():
    for c in CL.CLAIMS.values():
        print(f"\n== {c.id}: {c.title}\n   {c.source}  {c.url}\n   where: {c.where}\n   claim: {c.statement}")
        print(f"   builds on lessons {', '.join(c.builds_on)}; code: {'; '.join(c.code)}")
        print(f"   main path: {c.main_path}\n   free GPU: {c.free_gpu}\n   free CPU: {c.free_cpu}")
        print(f"   sound null: {c.null_result}\n   extension: {c.extension}")


def check(lab, pkg: Path):
    probs = PK.check_package(pkg)
    print("\ncourse checker:", "SOUND (no problems)" if not probs else f"{len(probs)} problem(s)")
    for p in probs:
        print("  ", p)
    P = PK.load(pkg)
    if P["claim"] and P["results"] is not None and P["claims"] is not None:
        dec = PK.recomputed_decisions(P["claim"], P["results"])
        cards = {p.parent.name: yaml.safe_load(p.read_text()) for p in (pkg / "runs").glob("*/run_card.yaml")}
        mine = lab.claim_problems(P["claims"], dec)
        for c in P["claim"]["comparisons"]:
            mine += lab.comparison_problems(P["claim"], c, cards)
        print("your checks:", mine or "no problems")
    return probs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claims", action="store_true")
    ap.add_argument("--check", type=Path, default=None)
    ap.add_argument("--variant", choices=["cpu", "t4", "main"], default="cpu")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", type=Path, default=Path("runs/m20/l201/capstone-qkclip"))
    a = ap.parse_args(argv)
    if a.claims:
        show_claims()
        return
    if a.print or a.variant != "cpu":
        print("\n".join(SC.print_commands(a.variant, a.seeds, str(a.out))))
        return
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    if a.check:
        sys.exit(1 if check(lab, a.check) else 0)
    t0 = time.perf_counter()
    res = SC.run(a.out, "cpu", a.seeds, interval=lab.hierarchical_interval, decide=lab.decide)
    print(f"\ncorrectness checks: {res['checks']}")
    print(f"tau = {res['tau']:g} (rule: round({SC.TAU_FRACTION} x seed-0 baseline's final max logit))")
    for c in res["comparisons"]:
        print(f"{c['name']:12s} {c['a']} - {c['b']}: {c['mean_diff']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]  "
              f"seed-t [{c['t_ci'][0]:+.4f}, {c['t_ci'][1]:+.4f}]  -> {c['decision']}")
    nf = res["noise_floor"]
    print(f"noise floor: seed std {nf['seed_std']:.4f}, MDE {nf['mde']:.4f} ({nf['n_seeds']} seeds)")
    print(f"\n{'run':16s} {'max logit':>9s} {'final':>7s} {'clipped':>8s} {'first clip':>10s} {'spikes':>6s} {'train s':>8s}")
    for n, s in res["secondary"].items():
        print(f"{n:16s} {s['max_logit']:9.2f} {s['final_max_logit']:7.2f} {s['clipped_head_updates']:8d} "
              f"{str(s['first_clip_step']):>10s} {s['loss_spikes']:6d} {s['train_seconds'] or 0:8.0f}")
    check(lab, a.out)
    print(f"\ntotal {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
