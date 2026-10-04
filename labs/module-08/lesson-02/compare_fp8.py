"""Lab 08.2, step 4: the FP8 arms against BF16, paired by seed and by window, and the contract's rule.

    python labs/module-08/lesson-02/compare_fp8.py                                  # runs/m08/l82/cpu
    python labs/module-08/lesson-02/compare_fp8.py --variant main --device cuda --speedup 1.31 1.27 1.35
    LAB_TARGET=solution python labs/module-08/lesson-02/compare_fp8.py              # use the reference decide()

For every arm and seed: the held-out loss on 256 fixed validation windows of 128 tokens, evaluated in the
precision the arm trained with. The gap to BF16 is computed per seed (same initial weights and data order), then
averaged over seeds window by window and bootstrapped over windows. The noise floor is the seed std of the BF16
arm. The margin is DeepSeek-V3's reported tolerance, 0.25% of the BF16 loss (section 3.3: "relative loss error ...
remains consistently below 0.25%"). Speed: pass ``--speedup point lo hi`` from bench_fp8.py on real kernels; the
emulated arms' tokens/s are printed only to show that emulation is slower, never as a speed result.
"""

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m08  # noqa: E402
from train_fp8 import VARIANTS, run_dir  # noqa: E402

from frontierlab.labkit import load_target  # noqa: E402
from frontierlab.stats import min_detectable_effect  # noqa: E402

lab = load_target(str(Path(__file__).parent / "test_lab.py"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--root", type=Path, default=Path("runs/m08/l82"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--speedup", type=float, nargs=3, default=None, metavar=("POINT", "LO", "HI"))
    a = ap.parse_args(argv)
    _, steps, _, _, seeds, arms, _ = VARIANTS[a.variant]
    losses = {}
    for arm in arms:
        for s in seeds:
            run = run_dir(a.root, a.variant, arm, s)
            if (run / "checkpoint.pt").exists() and m08.finished(run, steps):
                losses[(arm, s)] = m08.eval_losses(run, device=a.device)
    base = [m08.mean(losses[("bf16", s)]) for s in seeds if ("bf16", s) in losses]
    if len(base) < 1:
        raise SystemExit("train the bf16 arm first (train_fp8.py)")
    sd = statistics.stdev(base) if len(base) > 1 else float("nan")
    margin = 0.0025 * m08.mean(base)
    print(f"BF16 held-out loss per seed: {', '.join(f'{b:.4f}' for b in base)}; seed std {sd:.4f}; "
          f"MDE with {len(base)} seeds per arm {min_detectable_effect(sd, len(base)):.4f} (unpaired)")
    print(f"margin (0.25% of BF16 loss): {margin:.4f} nats\n")
    print(f"{'arm':15s} {'held-out':>9s} {'per-seed gap vs bf16':>28s} {'paired gap, seed-avg [95% CI]':>32s} "
          f"{'tok/s':>7s}  decision")
    for arm in arms:
        if arm == "bf16" or not all((arm, s) in losses for s in seeds):
            continue
        per_seed = [m08.mean(losses[(arm, s)]) - m08.mean(losses[("bf16", s)]) for s in seeds]
        avg_arm = [statistics.mean(v) for v in zip(*(losses[(arm, s)] for s in seeds))]
        avg_ref = [statistics.mean(v) for v in zip(*(losses[("bf16", s)] for s in seeds))]
        c = m08.compare(avg_arm, avg_ref)
        rows = [r for s in seeds for r in m08.train_rows(run_dir(a.root, a.variant, arm, s))
                if r["step"] > 20 and "tok_per_s" in r]
        tps = statistics.median(r["tok_per_s"] for r in rows) if rows else float("nan")
        sp = None if arm.startswith("fp8-") else (tuple(a.speedup[1:]) if a.speedup else None)
        verdict = lab.decide((c["lo"], c["hi"]), margin, sp, per_seed)
        print(f"{arm:15s} {statistics.mean(m08.mean(losses[(arm, s)]) for s in seeds):9.4f} "
              f"{', '.join(f'{d:+.4f}' for d in per_seed):>28s} {m08.fmt(c):>32s} {tps:7.0f}  {verdict}")
        log = m08.precision_rows(run_dir(a.root, a.variant, arm, seeds[0]))
        if log:
            last = log[-1]
            print(f"{'':15s} precision log (seed {seeds[0]}, step {last['step']}): rel err act {last.get('act_rel_err', 0):.4f} "
                  f"weight {last.get('weight_rel_err', 0):.4f} grad {last.get('grad_rel_err', 0):.4f}; "
                  f"grad underflow {last.get('grad_underflow', 0):.4f}; worst grad layer {last.get('worst_grad_layer')}")
    rows = [r for s in seeds for r in m08.train_rows(run_dir(a.root, a.variant, "bf16", s)) if r["step"] > 20]
    if rows:
        print(f"\nbf16 median tok/s {statistics.median(r['tok_per_s'] for r in rows):.0f}. Emulated arms (fp8-*) measure "
              "numerics only: their tok/s is the cost of emulation, not of FP8.")


if __name__ == "__main__":
    main()
