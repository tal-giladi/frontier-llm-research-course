"""Train the arms of lab 04.3 from the Module 4 base model, all at equal training tokens.

    python labs/module-04/lesson-03/extend_arms.py                       # free CPU: 256 -> 1024
    python labs/module-04/lesson-03/extend_arms.py --variant main --print  # show the main-path commands

Arms (each continues from the same base checkpoint with a fresh optimizer, seed 0, same token budget):

    ctrl-short    keep training at the original length (RoPE unchanged): what extra tokens alone do
    yarn-within   YaRN (s = new/original), windows inside long documents only
    yarn-random   YaRN, the loop's ordinary random windows (they cross document boundaries)
    pi-within     Position Interpolation, windows inside long documents only

Rerun the same command after an interruption: every arm resumes exactly (``frontierlab.longctx.extend``).
Then score them with ``compare_arms.py``.
"""

import argparse
import subprocess
import sys

VARIANTS = {
    # name: (base run, preset, original length, new length, batch at new length, steps, extra loop args)
    "cpu": ("runs/m04/base-cpu", "toy", 256, 1024, 4, 200, []),
    "t4": ("runs/m04/base-t4", "pilot-10m", 512, 2048, 8, 400, ["--device", "cuda"]),
    "main": ("runs/m01/main/seeds-lr3e-3-s0", "baseline0", 1024, 8192, 4, 2000,
             ["--device", "cuda", "--dtype", "bf16", "--grad-accum", "8", "--peak", "H100-SXM"]),
}


def arms(variant, base=None):
    default_base, preset, L0, L1, B1, steps, extra = VARIANTS[variant]
    base = base or default_base
    s = L1 / L0
    B0 = B1 * L1 // L0                                  # same tokens per step at the original length
    common = ["--preset", preset, "--steps", str(steps), "--lr", "1e-3", "--warmup", "20", "--seed", "0",
              "--eval-every", str(steps), "--log-every", str(max(1, steps // 10)), "--init-from",
              f"{base}/checkpoint.pt", *extra]
    yarn = ["--rope", "yarn", "--factor", f"{s:g}", "--original", str(L0)]
    return {
        "ctrl-short": ["--seq", str(L0), "--batch", str(B0), "--rope", "default"],
        "yarn-within": ["--seq", str(L1), "--batch", str(B1), *yarn, "--long-fraction", "1.0"],
        "yarn-random": ["--seq", str(L1), "--batch", str(B1), *yarn],
        "pi-within": ["--seq", str(L1), "--batch", str(B1), "--rope", "pi", "--factor", f"{s:g}", "--long-fraction", "1.0"],
    }, common


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--base", default=None, help="base run folder (main path: your Module 1 Baseline-0 seed-0 run, "
                    "runs/m01/main/seeds-lr<chosen lr>-s0)")
    ap.add_argument("--out", default="runs/m04")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    table, common = arms(a.variant, a.base)
    for name in a.arms or list(table):
        cmd = [sys.executable, "-m", "frontierlab.longctx.extend", "--run", f"{a.out}/{a.variant}-{name}",
               *table[name], *common, "--question", f"lab 04.3 arm {name}"]
        print(" ".join(cmd), flush=True)
        if not a.print and subprocess.call(cmd):
            sys.exit(f"arm {name} failed")


if __name__ == "__main__":
    main()
