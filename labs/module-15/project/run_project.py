"""Module 15 project: a budget-matched test-time-compute report with a recommendation for a latency target.

    python labs/module-15/project/run_project.py                         # free CPU, about 5 minutes after lab 15.1
    python labs/module-15/project/run_project.py --latency-x 1.2 4
    python labs/module-15/project/run_project.py --variant main --print

Runs lesson 15.1's experiment (``labs/module-15/lesson-01/ttc_lab.py``, your ``lab.py`` there) for two sampling
seeds on the same policy and verifiers, then applies the decision rule of ``frontierlab.ttc.report`` at each
budget and each latency target (a multiple of one greedy short chain's latency), and says whether the
recommendation is the same for both seeds.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from frontierlab.labkit import load_path
from frontierlab.ttc import report as R

HERE = Path(__file__).resolve().parent
L151 = HERE.parent / "lesson-01"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["cpu", "main", "t4"], default="cpu")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--latency-x", type=float, nargs="*", default=[1.2, 2.0, 4.0])
    ap.add_argument("--budgets", type=float, nargs="*", default=[40, 80, 160, 320])
    a = ap.parse_args(argv)
    if a.variant != "cpu":
        print("# Main path (1x H100; not run in this build; part of the Module 15 pilot): lesson 15.1's commands")
        print("python labs/module-15/lesson-01/ttc_lab.py --variant main --print")
        print("# then one report per latency target and two sampling seeds (rerun `sample` and `score` with --seed 1"
              " into runs/m15/project-main-s1):")
        for lat in (10, 20, 40):
            for s in (0, 1):
                print(f"python -m frontierlab.ttc.hf_ttc report --out runs/m15/project-main-s{s} --budget 8192 --latency {lat}")
        return
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    ttc_lab = load_path(str(L151 / "ttc_lab.py"))
    lab = load_path(str(L151 / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    results = {}
    for s in a.seeds:
        print(f"\n######## sampling seed {s}")
        results[s] = ttc_lab.run(lab, seed=s, latency_x=a.latency_x[0], quiet=(s != a.seeds[0]))
    print("\n######## recommendations (decision rule of frontierlab.ttc.report)")
    base_lat = {s: results[s]["latency_target"] / a.latency_x[0] for s in a.seeds}
    for x in a.latency_x:
        for B in a.budgets:
            picks = []
            for s in a.seeds:
                rec = R.recommend(results[s]["rows"], B, x * base_lat[s], seed=s)
                ch = rec["choice"]
                picks.append(ch["label"] if ch else "none")
                lab_ = f"{ch['label']} ({ch['success']:.3f}, {ch['pte']:.0f} pte, {ch['latency_s'] * 1e3:.1f} ms)" if ch else "none"
                print(f"  latency <= {x:g} x short chain, budget {B:4.0f}: seed {s}: {lab_}")
            print(f"      same choice for every seed: {len(set(picks)) == 1}")


if __name__ == "__main__":
    main()
