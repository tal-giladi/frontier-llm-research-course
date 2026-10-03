"""Lab 03.3: train the logit-control arms on Data-v0 (free CPU: toy preset; GPU: see --variant).

    python labs/module-03/lesson-03/train_arms.py                      # free CPU, 6 runs
    python labs/module-03/lesson-03/train_arms.py --variant t4         # free GPU (T4, fp32)
    python labs/module-03/lesson-03/train_arms.py --variant main       # main path (1x H100 or A100)

Arms (all branch from the same preset, same data order, same seed, same steps):
  b0, no-qknorm, sink, gated            at the default learning rate
  b0@hi, no-qknorm@hi                   at a raised learning rate (3.3x), where logit growth should show
Runs go to runs/m03/l33/<variant>/<arm>-s<seed>; rerunning skips finished runs and resumes
interrupted ones exactly (lesson 01.1).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_variant  # noqa: E402

VARIANTS = {   # preset, steps, batch, accum, seq, lr, extra loop args
    "cpu": ("toy", 300, 16, 1, 128, 3e-3, ["--eval-every", "300", "--ckpt-every", "100", "--eval-windows", "64"]),
    "t4": ("pilot-10m", 2000, 32, 1, 512, 3e-3, ["--dtype", "fp32", "--eval-every", "500", "--ckpt-every", "250"]),
    "main": ("pilot-30m", 4000, 32, 2, 1024, 3e-3, ["--dtype", "bf16", "--eval-every", "500", "--ckpt-every", "500",
                                                    "--peak", "H100-SXM"]),
}
ARMS = [("b0", 1.0), ("no-qknorm", 1.0), ("sink", 1.0), ("gated", 1.0), ("b0", 10 / 3), ("no-qknorm", 10 / 3)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None, help="arm names to run (e.g. b0 sink)")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/m03/l33"))
    a = ap.parse_args()
    preset, steps, batch, accum, seq, lr, extra = VARIANTS[a.variant]
    for arm, mult in ARMS:
        name = arm + ("@hi" if mult != 1.0 else "")
        if a.only and name not in a.only:
            continue
        run = a.out / a.variant / f"{name}-s{a.seed}"
        log = run / "metrics.jsonl"
        if log.exists() and f'"step": {steps},' in log.read_text() and '"val"' in log.read_text():
            print(f"{run}: finished, skipping")
            continue
        argv = ["--run", str(run), "--preset", preset, "--steps", str(steps), "--batch", str(batch),
                "--grad-accum", str(accum), "--seq", str(seq), "--lr", f"{lr * mult:g}", "--seed", str(a.seed),
                "--question", f"lesson 03.3: logit control, arm {name}", *extra]
        if a.max_minutes:
            argv += ["--max-minutes", str(a.max_minutes)]
        cfg, argv = train_variant.plan(arm, argv)
        print(f"\n=== {name}: {train_variant.m03.describe(cfg, seq)}")
        train_variant.train(cfg, argv)


if __name__ == "__main__":
    main()
