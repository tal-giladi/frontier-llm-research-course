"""Lab 08.3, step 3: FP4 training arms and the NVFP4 recipe's ingredients, removed one at a time.

    python labs/module-08/lesson-03/train_fp4.py                  # free CPU: toy, emulated FP4 (numerics only)
    python labs/module-08/lesson-03/train_fp4.py --variant t4     # free GPU (emulation on a T4)
    python labs/module-08/lesson-03/train_fp4.py --variant main --print   # B200 commands (Transformer Engine); not piloted

Arms (seed 0; bf16 and nvfp4 also seed 1 for the noise floor):
  bf16           no quantisation
  mxfp4          naive FP4: OCP MXFP4 (1 x 32, E8M0 scales) on every GEMM operand, round to nearest, no RHT
  nvfp4          the recipe of arXiv 2509.25149 section 4 (1 x 16 activations and gradients, 16 x 16 weights,
                 E4M3 block scales + FP32 tensor scale, RHT on Wgrad inputs, stochastic rounding on gradients),
                 every block quantised
  nvfp4-keep     the same with the last block in BF16 (1 of 4 blocks = 25% of the linears; the paper keeps
                 the first 2 and last 8 blocks of its 12B model, 16% of its linear layers)
  nvfp4-no-sr    ablation: gradients rounded to nearest
  nvfp4-no-rht   ablation: no Hadamard transform
Runs: runs/m08/l83/<variant>/<arm>-s<seed>. Exact resume: rerun after an interruption.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m08  # noqa: E402

VARIANTS = {
    "cpu": ("toy", 150, 16, 128, ["--eval-every", "1000", "--ckpt-every", "50", "--eval-windows", "64", "--log-every", "10",
                                   "--device", "cpu"], "last1"),
    "t4": ("pilot-10m", 1000, 16, 512, ["--dtype", "fp32", "--eval-every", "1000", "--ckpt-every", "100",
                                         "--eval-windows", "64", "--log-every", "20", "--device", "cuda", "--max-minutes", "50"],
           "last1"),
}
ARMS = {
    "bf16": ([], (0, 1)),
    "mxfp4": (["--recipe", "mxfp4"], (0,)),
    "nvfp4": (["--recipe", "nvfp4", "--precision-log", "--precision-every", "10"], (0, 1)),
    "nvfp4-keep": (["--recipe", "nvfp4"], (0,)),
    "nvfp4-no-sr": (["--recipe", "nvfp4-no-sr"], (0,)),
    "nvfp4-no-rht": (["--recipe", "nvfp4-no-rht"], (0,)),
}

MAIN_NOTE = """# Main path: 1x B200, NVIDIA Transformer Engine (NVFP4BlockScaling recipe; TE documents NVFP4 training on
# SM 10.0 / 10.3 only). NOT PILOTED: Colab offers no Blackwell GPU. This course does not ship a TE training loop; the
# commands below train the same arms with the course's emulation on a B200 (numerics only). Real NVFP4 speed needs
# TE's kernels; lesson 08.3 gives the PROJECTED bound (frontierlab.precision.cost) and NVIDIA's figures (company claim)."""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS) + ["main"], default="cpu")
    ap.add_argument("--root", type=Path, default=Path("runs/m08/l83"))
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    if a.variant == "main":
        print(MAIN_NOTE)
        preset, steps, B, T, extra, keep = ("pilot-70m", 4000, 64, 1024, ["--dtype", "bf16", "--device", "cuda",
                                            "--eval-every", "1000", "--ckpt-every", "500", "--log-every", "20"], "first1,last2")
    else:
        preset, steps, B, T, extra, keep = VARIANTS[a.variant]
    for arm, (args, seeds) in ARMS.items():
        for seed in seeds:
            run = a.root / a.variant / f"{arm}-s{seed}"
            argv_ = ["--preset", preset, "--batch", str(B), "--seq", str(T), "--seed", str(seed), *extra, *args]
            if arm == "nvfp4-keep":
                argv_ += ["--keep-high", keep]
            if a.print or a.variant == "main":
                print("python -m frontierlab.precision.train --run", run, "--steps", steps, " ".join(argv_))
                continue
            dt = m08.run_arm(run, argv_, steps)
            print(f"{run}: {dt / 60:.1f} min")


if __name__ == "__main__":
    main()
