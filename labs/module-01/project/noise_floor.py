"""Noise floor of Baseline-0 on Eval Suite v0, from the seed runs' eval_v0_val.json files.

    python labs/module-01/project/noise_floor.py runs/m01/cpu/seeds-*
    python labs/module-01/project/noise_floor.py runs/m01/main/seeds-* --out reports/m01-noise-floor.json

For each metric (held-out loss, LAMBADA target log-probability, LAMBADA accuracy) it prints:
  * the per-seed means, the seed std and the standard error of the seed mean;
  * the eval-sampling standard error within one run (item std / sqrt(items));
  * the minimum detectable effect (80% power, alpha 0.05) for 2, 3 and 5 seeds per arm;
  * an A/A check: the paired bootstrap of seed 1 vs seed 0 on the same items. Seed replicates are the
    same method, so this "difference" is pure noise; a CI that excludes zero shows why item-level
    intervals alone cannot support a claim about a method.
It refuses to run if the runs' cards are not seed replicates of each other (frontierlab.record).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from frontierlab.record import comparable, diff_cards
from frontierlab.stats import min_detectable_effect, paired_bootstrap, summary


def metrics(res: dict) -> dict[str, np.ndarray]:
    items = res["lambada"]["items"]
    return {"heldout_loss": np.asarray(res["heldout"]["losses"]),
            "lambada_logprob": np.asarray([it["logprob"] for it in items]),
            "lambada_acc": np.asarray([float(it["correct"]) for it in items])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    runs = sorted(a.runs)
    for r in runs[1:]:
        f = diff_cards(runs[0], r, seeds_are_replicates=True)
        if not comparable(f):
            bad = [str(x) for x in f if x.severity == "invalidates"]
            raise SystemExit(f"{r} is not a seed replicate of {runs[0]}:\n  " + "\n  ".join(bad))
    res = {r: json.loads((r / f"eval_v0_{a.split}.json").read_text()) for r in runs}
    pins = {(x["version"], x["lambada"]["revision"], x["heldout"]["windows"], x["heldout"]["T"]) for x in res.values()}
    if len(pins) != 1:
        raise SystemExit(f"runs were scored with different Eval v0 settings: {pins}")
    per_run = {r: metrics(x) for r, x in res.items()}
    report = {}
    for m in ("heldout_loss", "lambada_logprob", "lambada_acc"):
        means = [float(per_run[r][m].mean()) for r in runs]
        s = summary(means)
        within = float(np.mean([per_run[r][m].std(ddof=1) / math.sqrt(per_run[r][m].size) for r in runs]))
        mde = {n: min_detectable_effect(s["std"], n) for n in (2, 3, 5)}
        aa = paired_bootstrap(per_run[runs[1]][m], per_run[runs[0]][m]) if len(runs) > 1 else None
        report[m] = {"per_seed": means, "seed_std": s["std"], "seed_sem": s["sem"], "eval_se_within_run": within,
                     "mde": mde, "aa_paired_ci": aa["ci"] if aa else None}
        print(f"{m}")
        print("  per seed: " + "  ".join(f"{v:.4f}" for v in means))
        print(f"  seed std {s['std']:.4f}   SE of the seed mean {s['sem']:.4f}   eval-sampling SE within a run {within:.4f}")
        print("  MDE: " + "  ".join(f"{n} seeds {v:.4f}" for n, v in mde.items()))
        if aa:
            print(f"  A/A (seed {runs[1].name} - seed {runs[0].name}, same items): {aa['mean_diff']:+.4f} "
                  f"CI [{aa['ci'][0]:+.4f}, {aa['ci'][1]:+.4f}]")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"runs": [str(r) for r in runs], "metrics": report}, indent=2))
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
