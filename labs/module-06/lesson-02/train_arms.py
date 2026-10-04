"""Lab 06.2, step 3: Baseline-0 vs HC vs mHC, at the standard learning rate and at two raised ones.

    python labs/module-06/lesson-02/train_arms.py                       # free CPU (toy)
    python labs/module-06/lesson-02/train_arms.py --variant t4          # free GPU (T4, fp32, pilot-10m)
    python labs/module-06/lesson-02/train_arms.py --variant main        # main path (pilot-30m), not run in this build
    python labs/module-06/lesson-02/train_arms.py --variant main70      # ladder rung 2 (pilot-70m), not run in this build

Arms (same preset, data order and token budget; n = 4 streams; mHC with t_max = 20):
  b0, hc, mhc            at the variant's learning rate (1.5e-3), seeds 0 and 1 (the noise floor)
  b0, hc, mhc @ 1e-2     raised learning rate, seed 0
  b0, hc, mhc @ 3e-2     raised further, seed 0 (where instability is most likely to show at this scale)
Every run logs blocks.jsonl with the Amax gains of the HC/mHC residual maps every 20 steps (mHC section 3.1).
The 1.5e-3 runs are shared with lessons 06.1 (b0) and 06.3 (mhc).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m06  # noqa: E402

ARMS = ("b0", "hc", "mhc")
RAISED = (1e-2, 3e-2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(m06.VARIANTS), default="cpu")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    for seed in a.seeds:
        for arm in ARMS:
            m06.train_arm(a.variant, arm, seed, question=f"lesson 06.2: residual design, arm {arm}", max_minutes=a.max_minutes, device=a.device)
    for lr in RAISED:
        for arm in ARMS:
            m06.train_arm(a.variant, arm, 0, lr=lr,
                          question=f"lesson 06.2: residual design at raised lr {lr:g}, arm {arm}",
                          max_minutes=a.max_minutes, device=a.device)


if __name__ == "__main__":
    main()
