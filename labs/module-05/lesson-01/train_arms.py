"""Lab 05.1: train Baseline-0 and two 3:1 hybrids at equal parameters and equal tokens.

    python labs/module-05/lesson-01/train_arms.py                     # free CPU (toy, 256 tokens)
    python labs/module-05/lesson-01/train_arms.py --variant t4        # free GPU (T4, fp32)
    python labs/module-05/lesson-01/train_arms.py --variant main      # main path (1x H100 / A100), fla kernels
    python labs/module-05/lesson-01/train_arms.py --print             # show the commands, run nothing

Arms (same preset, same data order, same seed, same tokens; hybrids' SwiGLU width matched so non-embedding
parameters equal Baseline-0's):  b0,  hybrid-kda (3 KDA : 1 GQA),  hybrid-gdn (3 Gated DeltaNet : 1 GQA).
Runs go to runs/m05/l51/<variant>/<arm>-s<seed>; rerunning skips finished runs and resumes interrupted ones.
Then compare them with ``labs/module-05/heldout_compare.py`` (the command is printed at the end).
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = {   # preset, steps, batch, accum, seq, lr, wrapper args, extra loop args
    "cpu": ("toy", 600, 16, 1, 256, 3e-3, ["--chunk", "16"],
            ["--eval-every", "600", "--ckpt-every", "100", "--eval-windows", "64", "--log-every", "50"]),
    "t4": ("pilot-10m", 2000, 32, 1, 512, 3e-3, ["--chunk", "32"],
           ["--dtype", "fp32", "--eval-every", "500", "--ckpt-every", "250", "--max-minutes", "150"]),
    "main": ("pilot-30m", 4000, 32, 2, 1024, 3e-3, ["--chunk", "64", "--linear-mode", "fla"],
             ["--dtype", "bf16", "--eval-every", "500", "--ckpt-every", "500", "--peak", "H100-SXM"]),
}
ARMS = ("b0", "hybrid-kda", "hybrid-gdn")
EXTRA_ARMS = ("hybrid-kda-silu",)      # step 5: run with --only hybrid-kda-silu


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/m05/l51"))
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    preset, steps, batch, accum, seq, lr, wargs, extra = VARIANTS[a.variant]
    runs = []
    for arm in ARMS + EXTRA_ARMS:
        if (a.only and arm not in a.only) or (not a.only and arm in EXTRA_ARMS):
            continue
        run = a.out / a.variant / f"{arm}-s{a.seed}"
        runs.append(str(run))
        cmd = [sys.executable, str(HERE.parent / "train_arm.py"), "--arm", arm,
               *(["--match-params"] if arm != "b0" else []), *wargs, "--",
               "--run", str(run), "--preset", preset, "--steps", str(steps), "--batch", str(batch),
               "--grad-accum", str(accum), "--seq", str(seq), "--lr", str(lr), "--seed", str(a.seed),
               "--question", "05.1: does a 3:1 linear hybrid match Baseline-0 at equal parameters and tokens?", *extra]
        print(" ".join(cmd))
        if a.print:
            continue
        log = run / "metrics.jsonl"
        if log.exists() and any(f'"step": {steps}' in line and '"val"' in line for line in log.read_text().splitlines()):
            print(f"  {run} finished, skipping")
            continue
        if subprocess.call(cmd) != 0:
            sys.exit(f"run {run} failed")
    print("\nthen:\n  python labs/module-05/heldout_compare.py " + " ".join(runs) + f" --seq {seq} --margin 0.03")


if __name__ == "__main__":
    main()
