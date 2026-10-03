"""Train the Module 4 base model: Baseline-0's recipe at a short context, the model every Module 4 lab extends.

    python labs/module-04/lesson-01/train_base.py                       # free CPU: toy, 256 tokens
    python labs/module-04/lesson-01/train_base.py --variant t4          # free GPU (T4)
    python labs/module-04/lesson-01/train_base.py --print               # show the command, run nothing

The main path does not train a base here: it uses the Module 1 project's Baseline-0 seed-0 run
(trained at 1,024 tokens). This script only wraps ``frontierlab.train.loop`` with fixed arguments so
every learner's base is the same run; rerun the same command after an interruption to resume it.

Measured on the build laptop (16 threads, torch 2.14.1 CPU, 2026-10-03, other jobs running): the CPU
variant takes about 30 minutes (1,500 steps of 16 x 256 tokens, about 3,300 tokens/s).
"""

import argparse
import subprocess
import sys

VARIANTS = {
    "cpu": ["--preset", "toy", "--seq", "256", "--batch", "16", "--steps", "1500", "--lr", "3e-3", "--warmup", "50"],
    "t4": ["--preset", "pilot-10m", "--seq", "512", "--batch", "32", "--steps", "3000", "--lr", "3e-3", "--warmup", "100",
           "--device", "cuda", "--max-minutes", "150"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--run", default=None, help="run folder (default runs/m04/base-<variant>)")
    ap.add_argument("--print", action="store_true", help="print the command and exit")
    a = ap.parse_args()
    run = a.run or f"runs/m04/base-{a.variant}"
    cmd = [sys.executable, "-m", "frontierlab.train.loop", "--run", run, *VARIANTS[a.variant],
           "--eval-every", "500", "--log-every", "100", "--question", "Module 4 base model (short context)"]
    print(" ".join(cmd))
    if not a.print:
        sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
