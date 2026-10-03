"""Noise floor, MDE and paired vs unpaired comparisons from run_seeds.py outputs (uses YOUR lab.py).

    python labs/module-01/lesson-04/analyze.py
    LAB_TARGET=solution python labs/module-01/lesson-04/analyze.py
    python labs/module-01/lesson-04/analyze.py --out runs/l14-gpu
"""

import argparse
import json
from pathlib import Path

import numpy as np

from frontierlab.labkit import load_target
from frontierlab.stats import holm, paired_bootstrap, summary

lab = load_target(str(Path(__file__).parent / "test_lab.py"))


def load(out: Path, arm: str) -> dict[int, np.ndarray]:
    runs = {}
    for p in sorted(out.glob(f"{arm}-s*/window_losses.json")):
        runs[int(p.parent.name.rsplit("-s", 1)[1])] = np.asarray(json.loads(p.read_text()))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/l14-seeds"))
    ap.add_argument("--effect", type=float, default=0.02, help="effect size you want to detect (nats)")
    a = ap.parse_args()
    base, new = load(a.out, "base"), load(a.out, "lr4e-3")
    bm = {s: float(v.mean()) for s, v in base.items()}
    nm = {s: float(v.mean()) for s, v in new.items()}
    sb = summary(list(bm.values()))
    print("1. Noise floor (base arm, one mean val loss per seed)")
    print("   " + "  ".join(f"s{s} {m:.4f}" for s, m in bm.items()))
    print(f"   mean {sb['mean']:.4f}  seed std {sb['std']:.4f}  (n = {sb['n']})")
    print("\n2. Minimum detectable effect (80% power, alpha 0.05, two independent arms)")
    for n in (2, 3, 5):
        print(f"   {n} seeds per arm: MDE = {lab.mde(sb['std'], n):.4f} nats")
    print(f"   seeds per arm to detect {a.effect}: {lab.seeds_needed(sb['std'], a.effect)}")
    s0 = min(set(base) & set(new))
    pw = paired_bootstrap(new[s0], base[s0])
    uw = lab.unpaired_bootstrap(new[s0], base[s0])
    print(f"\n3. One seed pair (seed {s0}), 256 windows, lr4e-3 minus base")
    print(f"   window std of base losses {base[s0].std(ddof=1):.3f}; std of paired differences "
          f"{(new[s0] - base[s0]).std(ddof=1):.4f}")
    print(f"   unpaired bootstrap: {uw['mean_diff']:+.4f}  CI [{uw['ci'][0]:+.4f}, {uw['ci'][1]:+.4f}]")
    print(f"   paired bootstrap:   {pw['mean_diff']:+.4f}  CI [{pw['ci'][0]:+.4f}, {pw['ci'][1]:+.4f}]")
    print("   (both only measure eval-sampling noise for THESE two trained models, not seed noise)")
    print("\n4. Paired window bootstrap for every seed pair: does the sign hold across seeds?")
    pvals = []
    for s in sorted(set(base) & set(new)):
        r = paired_bootstrap(new[s], base[s])
        p_two = 2 * min(r["p_le_zero"], 1 - r["p_le_zero"])
        pvals.append(max(p_two, 1e-4))
        print(f"   seed {s}: {r['mean_diff']:+.4f}  CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
    shared = sorted(set(base) & set(new))
    s1 = sorted(base)[1]
    aa = paired_bootstrap(base[s1], base[s0])
    print(f"   A/A, base seed {s1} - base seed {s0} (the same method): {aa['mean_diff']:+.4f}  "
          f"CI [{aa['ci'][0]:+.4f}, {aa['ci'][1]:+.4f}]")
    print("\n5. Seed-level intervals (one number per run): the claim about the METHOD")
    sp = lab.seed_level_ci([bm[s] for s in shared], [nm[s] for s in shared], paired=True)
    su = lab.seed_level_ci(list(bm.values()), list(nm.values()), paired=False)
    print(f"   paired by seed ({len(shared)} pairs): {sp['mean_diff']:+.4f}  CI [{sp['ci'][0]:+.4f}, {sp['ci'][1]:+.4f}]")
    print(f"   unpaired Welch ({len(bm)} vs {len(nm)} runs): {su['mean_diff']:+.4f}  CI [{su['ci'][0]:+.4f}, {su['ci'][1]:+.4f}]")
    rho = np.corrcoef([bm[s] for s in shared], [nm[s] for s in shared])[0, 1]
    print(f"   correlation of per-seed losses across arms: {rho:+.2f}")
    print("\n6. Holm over the per-seed window tests above (as if each were a separate claim)")
    print("   raw p " + " ".join(f"{p:.4f}" for p in pvals) + "   Holm " + " ".join(f"{p:.4f}" for p in holm(pvals)))
    (a.out / "analysis.json").write_text(json.dumps({"base_seed_means": bm, "new_seed_means": nm,
                                                     "seed_std": sb["std"], "paired_seed_ci": sp["ci"],
                                                     "welch_ci": su["ci"]}, indent=2))


if __name__ == "__main__":
    main()
