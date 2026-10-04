"""Lab 06.3: a 2 x 2 factorial of two Module 6 changes - MTP (A) and the mHC residual (B) - and its analysis.

    python labs/module-06/lesson-03/factorial.py                  # free CPU: trains the missing cells, then analyses
    python labs/module-06/lesson-03/factorial.py --analyse-only
    python labs/module-06/lesson-03/factorial.py --variant main --device cuda     # main path, not run in this build

Cells (seeds 0 and 1, the variant's steps, equal tokens): b0 (neither), mtp-ds (A), mhc (B), mtp-ds+mhc (both).
Three of the four cells are the runs of lessons 06.1 and 06.2 (same command, same run directory); only the
combined cell is new. Analysis (lesson 06.3): cell means per seed, main effects, the interaction contrast with a
paired 95% bootstrap interval over the 256 windows (seed-averaged) and per seed, the additive prediction for
the combined cell against its measured loss, the "best of each branch" pick, and the cost of every cell
(training FLOPs from ``frontierlab.blocks.accounting``, measured wall-clock). Finally the integration budget:
how many runs a full factorial and a factorial-lite design need for the Lineage-F components.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m06  # noqa: E402
from frontierlab.labkit import load_path  # noqa: E402

CELLS = {(0, 0): "b0", (1, 0): "mtp-ds", (0, 1): "mhc", (1, 1): "mtp-ds+mhc"}
LINEAGE_F = ["mla", "mtp-ds", "mhc", "moe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(m06.VARIANTS), default="cpu")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=None)
    a = ap.parse_args()
    import os
    lab = load_path(str(Path(__file__).parent / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    if not a.analyse_only:
        for seed in a.seeds:
            for arm in CELLS.values():
                m06.train_arm(a.variant, arm, seed, question=f"lesson 06.3: 2x2 factorial MTP x mHC, cell {arm}",
                              max_minutes=a.max_minutes, device=a.device)
    v = m06.VARIANTS[a.variant]
    kw = dict(n=v["eval_n"], T=v["eval_T"], device=a.device)
    vocab = m06.vocab_size()
    per_seed = {c: [float(np.mean(m06.eval_losses(m06.run_dir(a.variant, arm, s), **kw))) for s in a.seeds]
                for c, arm in CELLS.items()}
    per_window = {c: m06.seed_mean_losses([m06.run_dir(a.variant, arm, s) for s in a.seeds], **kw)
                  for c, arm in CELLS.items()}
    means = {c: float(np.mean(x)) for c, x in per_seed.items()}
    print(f"{'cell':12s} {'A':>2s} {'B':>2s} {'per seed':>18s} {'mean':>8s} {'train MFLOP/tok':>16s} {'wall s':>7s}")
    for c, arm in CELLS.items():
        wc = np.mean([m06.wallclock(m06.run_dir(a.variant, arm, s)) for s in a.seeds])
        print(f"{arm:12s} {c[0]:2d} {c[1]:2d} {' '.join(f'{x:.4f}' for x in per_seed[c]):>18s} {means[c]:8.4f} "
              f"{m06.flops_per_token(arm, a.variant, vocab) / 1e6:16.3f} {wc:7.0f}")
    effA = m06.compare(per_window[(1, 0)], per_window[(0, 0)])
    effB = m06.compare(per_window[(0, 1)], per_window[(0, 0)])
    effA_withB = m06.compare(per_window[(1, 1)], per_window[(0, 1)])
    print(f"\neffect of A (MTP) without B: {m06.fmt(effA)};  with B on: {m06.fmt(effA_withB)}")
    print(f"effect of B (mHC) without A: {m06.fmt(effB)}")
    point, lo, hi = lab.interaction_ci(per_window)
    seeds_int = [lab.interaction({c: per_seed[c][i] for c in CELLS}) for i in range(len(a.seeds))]
    print(f"interaction (m11 - m10 - m01 + m00): {point:+.4f} [{lo:+.4f}, {hi:+.4f}]  per seed "
          f"{' '.join(f'{x:+.4f}' for x in seeds_int)}")
    nf = m06.noise_floor(per_seed[(0, 0)], len(a.seeds))
    print(f"noise floor: b0 seed std {nf['seed_std']:.4f}; an interaction contrast has 4 cells, so its seed-level "
          f"std is about 2x a single cell's: {2 * nf['seed_std']:.4f}")
    pred = lab.additive_prediction(means[(0, 0)], {"A": means[(1, 0)], "B": means[(0, 1)]})
    print(f"\nadditive prediction for A+B: {pred:.4f}; measured {means[(1, 1)]:.4f}; difference {means[(1, 1)] - pred:+.4f} "
          "(= the interaction)")
    best = {"A": means[(1, 0)] < means[(0, 0)], "B": means[(0, 1)] < means[(0, 0)]}
    picked = [k for k, keep in best.items() if keep]
    print(f"'best of each branch' keeps {picked or 'nothing'} (each beat b0 on its own at equal tokens); "
          f"the cheapest cell within the noise of the best is the defensible pick - see the lesson")
    print("\nIntegration budget for the Lineage-F components", LINEAGE_F)
    for design in ("full", "lite"):
        cells = lab.design_cells(LINEAGE_F, design)
        print(f"  {design:4s}: {len(cells):2d} cells x {len(a.seeds)} seeds = {len(cells) * len(a.seeds)} runs")


if __name__ == "__main__":
    main()
