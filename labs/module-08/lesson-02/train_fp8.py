"""Lab 08.2, step 3: train the same model in BF16 and in FP8, same seeds, same data, same tokens.

    python labs/module-08/lesson-02/train_fp8.py                    # free CPU: toy, emulated FP8 (numerics only)
    python labs/module-08/lesson-02/train_fp8.py --variant l4       # Colab L4 (sm89, real FP8): pilot, not run in this build
    python labs/module-08/lesson-02/train_fp8.py --variant main     # 1x H100: main path, not run in this build
    python labs/module-08/lesson-02/train_fp8.py --variant main --print   # only print the commands

Arms (each with seeds 0, 1 on CPU and L4; 0, 1, 2 on the main path):
  bf16           the baseline: no quantisation (fp32 on CPU, bf16 autocast on GPU)
  fp8-tensorwise emulated torchao "tensorwise" numerics (E4M3 forward, E5M2 gradients, per-tensor scales)  [CPU]
  fp8-rowwise    emulated torchao "rowwise" numerics (E4M3, per-row power-of-2 scales)                    [CPU]
  fp8-deepseek   emulated DeepSeek-V3 recipe (1x128 tiles, 128x128 weight blocks, E4M3 everywhere)         [all]
  ao-tensorwise  real torchao Float8 "tensorwise" kernels, torch.compile                                   [GPU]
  ao-rowwise     real torchao Float8 "rowwise" kernels, torch.compile                                      [GPU]
Arms at the same seed share initial weights and the data order exactly (the precision swap draws no random
numbers), so their difference is paired by seed and by evaluation window. Runs: runs/m08/l82/<variant>/<arm>-s<seed>.
Exact resume: rerun the same command after an interruption.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m08  # noqa: E402

VARIANTS = {   # preset, steps, batch, seq, seeds, arms, extra loop args
    "cpu": ("toy", 200, 16, 128, (0, 1), ("bf16", "fp8-tensorwise", "fp8-rowwise", "fp8-deepseek"),
            ["--eval-every", "1000", "--ckpt-every", "100", "--eval-windows", "64", "--log-every", "10", "--device", "cpu"]),
    "l4": ("pilot-30m", 2000, 32, 512, (0, 1), ("bf16", "ao-tensorwise", "ao-rowwise", "fp8-deepseek"),
           ["--dtype", "bf16", "--eval-every", "1000", "--ckpt-every", "250", "--eval-windows", "64", "--log-every", "20",
            "--device", "cuda", "--peak", "L4", "--max-minutes", "50"]),
    "main": ("pilot-70m", 4000, 64, 1024, (0, 1, 2), ("bf16", "ao-tensorwise", "ao-rowwise", "fp8-deepseek"),
             ["--dtype", "bf16", "--eval-every", "1000", "--ckpt-every", "500", "--eval-windows", "64", "--log-every", "20",
              "--device", "cuda", "--peak", "H100-SXM"]),
}
ARMS = {
    "bf16": [],
    "fp8-tensorwise": ["--recipe", "fp8-tensorwise"],
    "fp8-rowwise": ["--recipe", "fp8-rowwise"],
    "fp8-deepseek": ["--recipe", "fp8-deepseek", "--precision-log", "--precision-every", "20"],
    "ao-tensorwise": ["--torchao", "tensorwise", "--compile"],
    "ao-rowwise": ["--torchao", "rowwise", "--compile"],
}
COMPILED = {"bf16"}          # on GPU the baseline is compiled too, so the speed comparison is like for like


def arm_args(variant: str, arm: str, seed: int) -> list[str]:
    preset, _, batch, seq, _, _, extra = VARIANTS[variant]
    args = ["--preset", preset, "--batch", str(batch), "--seq", str(seq), "--seed", str(seed), *extra, *ARMS[arm]]
    if variant != "cpu" and arm in COMPILED:
        args.append("--compile")
    return args


def run_dir(root: Path, variant: str, arm: str, seed: int) -> Path:
    return root / variant / f"{arm}-s{seed}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--root", type=Path, default=Path("runs/m08/l82"))
    ap.add_argument("--data", default=None, help="Data-v0 folder (main path: the vocabulary-32768 build)")
    ap.add_argument("--print", action="store_true", help="print the commands instead of running them")
    a = ap.parse_args(argv)
    _, steps, _, _, seeds, arms, _ = VARIANTS[a.variant]
    for seed in seeds:
        for arm in arms:
            run = run_dir(a.root, a.variant, arm, seed)
            args = arm_args(a.variant, arm, seed) + (["--data", a.data] if a.data else [])
            if a.print:
                print("python -m frontierlab.precision.train --run", run, "--steps", steps, " ".join(args))
                continue
            dt = m08.run_arm(run, args, steps)
            print(f"{run}: {dt / 60:.1f} min")


if __name__ == "__main__":
    main()
