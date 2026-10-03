"""Train the toy preset with several seeds in two arms and keep every per-window validation loss.

    python labs/module-01/lesson-04/run_seeds.py                     # ~15 min on a 16-thread laptop
    python labs/module-01/lesson-04/run_seeds.py --steps 150         # ~8 min, a noisier noise floor
    python labs/module-01/lesson-04/run_seeds.py --preset pilot-10m --device cuda --batch 64 --seq 512 \
        --steps 2000 --out runs/l14-gpu                              # GPU (main path pilot)

Arms: ``base`` (lr 3e-3, seeds 0-4) and ``lr4e-3`` (lr 4e-3, seeds 0-2). Seed i sets the model
initialisation AND the data order (frontierlab.train.loop --seed), so base seed i and lr4e-3 seed i
see the same batches from the same starting weights: that is what makes them a matched pair.
Each run is scored on the same 256 fixed validation windows (window seed 1234); the losses go to
``<out>/<arm>-s<seed>/window_losses.json``. Then run analyze.py.
"""

import argparse
import json
from pathlib import Path

from frontierlab.data.loader import TokenData
from frontierlab.evals.heldout import window_losses
from frontierlab.train import loop

ARMS = {"base": {"lr": 3e-3, "seeds": [0, 1, 2, 3, 4]}, "lr4e-3": {"lr": 4e-3, "seeds": [0, 1, 2]}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="toy")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("runs/l14-seeds"))
    a = ap.parse_args()
    val = TokenData("val")
    for arm, spec in ARMS.items():
        for seed in spec["seeds"]:
            run = a.out / f"{arm}-s{seed}"
            if (run / "window_losses.json").exists():
                print(f"{run} done, skipping")
                continue
            args = ["--run", str(run), "--preset", a.preset, "--steps", str(a.steps), "--batch", str(a.batch),
                    "--seq", str(a.seq), "--lr", str(spec["lr"]), "--warmup", str(max(1, a.steps // 6)),
                    "--seed", str(seed), "--eval-every", str(10**9), "--log-every", "50",
                    "--ckpt-every", str(10**9), "--device", a.device,
                    "--question", "lesson 01.4: seed noise floor and a paired comparison"]
            if a.device.startswith("cuda"):
                args += ["--dtype", "bf16"]
            model = loop.main(args)
            losses = window_losses(model, val, a.windows, a.seq, device=a.device)
            (run / "window_losses.json").write_text(json.dumps(losses))
            print(f"{arm} seed {seed}: mean val loss {sum(losses) / len(losses):.4f}")


if __name__ == "__main__":
    main()
