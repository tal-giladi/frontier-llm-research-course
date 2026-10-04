"""A colleague's version of the Lineage-F report stage (the project's debugging task). It contains planted bugs.

    python labs/module-06/project/buggy_report.py                 # runs/m06/cpu, after run_project.py

It prints "the combination beats Baseline-0 and its best single branch, adopt it". Find out what each printed
comparison actually compares, check it with the run cards (python -m frontierlab.record) and by recomputing a
number by hand, fix it, and say whether the conclusion survives.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m06  # noqa: E402

COMBO = "mla+mtp-ds+mhc+moe"
SINGLES = ("mla", "mtp-ds", "mhc", "moe")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="cpu")
    a = ap.parse_args()
    v = m06.VARIANTS[a.variant]
    kw = dict(n=v["eval_n"], T=v["eval_T"], split="test")
    combo = np.mean([m06.eval_losses(m06.run_dir(a.variant, COMBO, s), **kw) for s in (0, 1)], axis=0)
    base = np.array(m06.eval_losses(m06.run_dir(a.variant, "b0", 0), **kw))          # "Baseline-0 at the same budget"
    singles = {c: np.array(m06.eval_losses(m06.run_dir(a.variant, c, 0), **kw)) for c in SINGLES}
    best = min(singles, key=lambda c: singles[c].mean())                              # best branch, by its test score
    print(f"combination {combo.mean():.4f}   Baseline-0 {base.mean():.4f}   best single ({best}) {singles[best].mean():.4f}")
    r1, r2 = m06.compare(combo, base), m06.compare(combo, singles[best])
    print(f"combination - Baseline-0 (same budget):  {m06.fmt(r1)}")
    print(f"combination - best single branch:        {m06.fmt(r2)}")
    if r1["ci"][1] < 0 and r2["ci"][1] < 0:
        print("Conclusion: the combination beats Baseline-0 and its best single branch at the same budget. Adopt it.")
    else:
        print("Conclusion: no clear winner.")


if __name__ == "__main__":
    main()
