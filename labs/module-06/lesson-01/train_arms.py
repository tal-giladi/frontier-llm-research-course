"""Lab 06.1, step 3: Baseline-0 vs multi-token prediction at equal tokens and at equal training FLOPs.

    python labs/module-06/lesson-01/train_arms.py                       # free CPU (toy), seeds 0 1
    python labs/module-06/lesson-01/train_arms.py --variant t4          # free GPU (T4, fp32, pilot-10m)
    python labs/module-06/lesson-01/train_arms.py --variant main        # main path (pilot-30m, 1x H100/A100), not run in this build
    python labs/module-06/lesson-01/train_arms.py --variant main70      # pilot ladder rung 2 (pilot-70m), not run in this build

Arms (same preset, data order, seeds, learning rate and schedule):
  b0          Baseline-0, next-token loss only, the variant's steps
  mtp-ds      DeepSeek-V3 MTP, D = 1, lambda 0.3 then 0.1 after 10/14.8 of the run (V3 section 4.2)
  mtp-meta    Meta's parallel heads, n = 2 (one extra head on the trunk), equal weight (Meta Eq. 2)
  b0@N steps  Baseline-0 trained for the steps whose training FLOPs equal mtp-ds's (the equal-FLOPs arm)
Runs go to runs/m06/<variant>/<arm>/s<seed>; lessons 06.3 and the project reuse them.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m06  # noqa: E402

ARMS = ("b0", "mtp-ds", "mtp-meta")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(m06.VARIANTS), default="cpu")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--print-only", action="store_true", help="print the plan (arms, steps, FLOPs) and stop")
    a = ap.parse_args()
    vocab = m06.vocab_size()
    eq = m06.equal_flops_steps("mtp-ds", "b0", a.variant, vocab)
    for arm in ARMS:
        print(f"{arm:10s} {m06.flops_per_token(arm, a.variant, vocab) / 1e6:9.3f} MFLOP/token (training)")
    print(f"equal-FLOPs Baseline-0: {eq} steps instead of {m06.VARIANTS[a.variant]['steps']}")
    if a.print_only:
        return
    for seed in a.seeds:
        for arm in ARMS:
            m06.train_arm(a.variant, arm, seed, question=f"lesson 06.1: MTP vs next-token, arm {arm}",
                          max_minutes=a.max_minutes, device=a.device)
        m06.train_arm(a.variant, "b0", seed, steps=eq, question="lesson 06.1: Baseline-0 at mtp-ds's training FLOPs",
                      max_minutes=a.max_minutes, device=a.device)


if __name__ == "__main__":
    main()
