"""Debugging task of the Module 11 project: a colleague's scaling-law prediction for the lesson 11.3 target.

    python labs/module-11/project/buggy_fit.py            # the colleague's pipeline and its prediction
    python labs/module-11/project/buggy_fit.py --fixed    # only after your diagnosis

Needs the lesson 11.1 iso-FLOP ladder (``runs/m11/l111/cpu/results.json``) and, for the comparison with a measured
loss, the lesson 11.3 target run. No training: a few seconds.

The colleague's summary: "I had only 9 usable ladder runs, so I added the evaluations logged at 30%, 50% and 70% of each run
as extra data points (36 points instead of 9). The fit is excellent in-sample, and when the 11.3 target finished, my
prediction was much closer to the measured loss than the pre-registered one. We should use my pipeline for Recipe-R."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

from frontierlab.scaling import derisk, ladder  # noqa: E402
from frontierlab.scaling import fit as sfit  # noqa: E402

FRACS = [0.3, 0.5, 0.7, 1.0]


def points(runs, fixed: bool):
    """The colleague's data set: every run at several fractions, as if each were a finished run of f·D tokens."""
    rows = []
    for r in runs:
        for f in ([1.0] if fixed else FRACS):
            v = derisk.loss_at_fraction(r["val_curve"], r["steps"], f)
            key = "N_total" if fixed else "N_nonemb"
            rows.append((r[key], f * r["D"], v))
    return np.array(rows, dtype=float)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixed", action="store_true")
    ap.add_argument("--variant", default="cpu")
    a = ap.parse_args(argv)
    res = json.loads((common.RUNS / "l111" / a.variant / "results.json").read_text())
    runs = [r for r in res["runs"] if r["D"] / r["N_total"] >= res["min_tpp"]]     # 11.1's rule, kept by both pipelines
    pre = common.RUNS / "l113" / a.variant / "prereg.json"
    if not pre.exists():
        raise SystemExit(f"{pre} missing: run lesson 11.3's `derisk_lab.py predict` first")
    t = derisk.load_prereg(pre)["target"]
    P = points(runs, a.fixed)
    fit = sfit.fit_parametric(P[:, 0], P[:, 1], P[:, 2])
    boots = sfit.bootstrap(P[:, 0], P[:, 1], P[:, 2], n_boot=200, seed=0)
    N_t = t["N_total"]                                   # the target's size, as written in the pre-registration
    mid, lo, hi = sfit.prediction_interval(boots, N_t, t["tokens"], 0.9)
    point = float(sfit.predict(fit, N_t, t["tokens"]))
    label = "fixed pipeline" if a.fixed else "colleague's pipeline"
    print(f"{label}: {len(P)} points; fit E {fit['E']:.3f}, A {fit['A']:.4g}, B {fit['B']:.4g}, alpha {fit['alpha']:.3f}, "
          f"beta {fit['beta']:.3f}; in-sample RMS(log) {fit['rms_log']:.4f}")
    print(f"prediction for {t['preset']} (N_total {N_t:,}, D {t['tokens']:,}): {point:.4f}, 90% interval [{lo:.4f}, {hi:.4f}]")
    run = common.RUNS / "l113" / a.variant / f"target-{t['preset']}"
    if (run / "metrics.jsonl").exists():
        r = ladder.read_run(run)
        if r["finished"]:
            print(f"measured (lesson 11.3 target run): {r['loss']:.4f}; error {point - r['loss']:+.4f}")


if __name__ == "__main__":
    main()
