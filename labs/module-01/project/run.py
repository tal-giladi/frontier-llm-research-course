"""Module 1 project: train Baseline-0 (learning-rate sweep, then seeds) with one fixed set of arguments.

    python labs/module-01/project/run.py --variant cpu  --stage sweep      # free CPU, ~4 min
    python labs/module-01/project/run.py --variant cpu  --stage seeds --lr 1.5e-3   # ~16 min
    python labs/module-01/project/run.py --variant t4   --stage sweep      # free GPU (Colab/Kaggle T4)
    python labs/module-01/project/run.py --variant main --stage seeds --lr <chosen>   # 1x H100

Every run goes to ``runs/m01/<variant>/<stage>-lr<lr>-s<seed>`` and is a normal
``frontierlab.train.loop`` run: it writes a run card, logs JSONL and checkpoints, and **resumes
exactly** if you run the same command again (for example after a Colab disconnect; on Colab add
``--max-minutes`` so the session checkpoints before it is cut off). Finished runs are skipped.

Stages:
  sweep  one seed (0), learning rates {0.5x, 1x, 2x} of the variant's default, SHORT runs
         (25% of the full length) - choose on VALIDATION loss, never on test.
  seeds  the chosen learning rate, seeds 0, 1, 2, full length.

Main path (projected, pending the Module 1 pilot): baseline0, 9,500 steps x 32 x 8 x 1,024 tokens =
2.49B tokens; 1.96e18 training FLOPs per run (frontierlab.flops); at an assumed 30% MFU on one H100
SXM (989e12 FLOP/s dense BF16) that is 1.84 GPU-hours per run, 5.5 for three seeds, plus the sweep
(3 x 25% = 0.75 runs, 1.4 GPU-hours).
"""

import argparse
from pathlib import Path

from frontierlab.train import loop

VARIANTS = {
    # preset, steps, batch, grad_accum, seq, default lr, extra args
    "main": ("baseline0", 9500, 32, 8, 1024, 3e-3, ["--dtype", "bf16", "--compile", "--peak", "H100-SXM",
                                                   "--eval-every", "500", "--ckpt-every", "500"]),
    "t4": ("pilot-10m", 4000, 32, 2, 512, 3e-3, ["--dtype", "fp32", "--eval-every", "500", "--ckpt-every", "250"]),
    "cpu": ("toy", 600, 16, 1, 128, 3e-3, ["--eval-every", "300", "--ckpt-every", "100"]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--stage", choices=["sweep", "seeds"], required=True)
    ap.add_argument("--lr", type=float, default=None, help="seeds stage: the learning rate chosen on validation")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--max-minutes", type=float, default=None, help="per-session limit (Colab)")
    ap.add_argument("--out", type=Path, default=Path("runs/m01"))
    a = ap.parse_args()
    preset, steps, batch, accum, seq, lr0, extra = VARIANTS[a.variant]
    if a.stage == "sweep":
        plan = [(lr, 0, steps // 4) for lr in (lr0 / 2, lr0, lr0 * 2)]
    else:
        if a.lr is None:
            raise SystemExit("--stage seeds needs --lr (the value you chose from the sweep on validation)")
        plan = [(a.lr, s, steps) for s in a.seeds]
    for lr, seed, n in plan:
        run = a.out / a.variant / f"{a.stage}-lr{lr:g}-s{seed}"
        ck = run / "metrics.jsonl"
        if ck.exists() and any(f'"step": {n},' in line and '"split": "val"' in line for line in ck.read_text().splitlines()):
            print(f"{run}: finished, skipping")
            continue
        args = ["--run", str(run), "--preset", preset, "--steps", str(n), "--batch", str(batch),
                "--grad-accum", str(accum), "--seq", str(seq), "--lr", str(lr), "--warmup", str(max(10, n // 20)),
                "--seed", str(seed), "--eval-windows", "64", "--log-every", "20",
                "--question", f"Module 1 project: Baseline-0 {a.stage} ({a.variant})", *extra]
        if a.max_minutes:
            args += ["--max-minutes", str(a.max_minutes)]
        loop.main(args)


if __name__ == "__main__":
    main()
