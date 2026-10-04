"""Lab 07.4, step 2: one stable run, decay branches off it, cosine runs for comparison, and continued training.

    python labs/module-07/lesson-04/run_schedules.py                 # free CPU (toy preset), about 40 minutes
    python labs/module-07/lesson-04/run_schedules.py --variant main  # main path, not run in this build
    python labs/module-07/lesson-04/run_schedules.py --print         # show the commands only

Arms (toy preset, AdamW = the loop's own optimizer, peak lr 3e-3, warmup 30, same seed and data order):
  stable                constant lr after warmup, trained in segments; its checkpoint is copied at each S0
  wsd-T300-d10          branch at S0 = 270, linear decay over 30 steps     (total 300)
  wsd-T600-d10          branch at S0 = 540, linear decay over 60 steps     (total 600)
  wsd-T600-d20          branch at S0 = 480, linear decay over 120 steps    (total 600)
  wsd-T600-d10-sqrt     branch at S0 = 540, 1-sqrt decay over 60 steps     (total 600)
  cos-T300, cos-T600    the loop's cosine schedule, a separate run per length
  wsd-T900-d10          continued training: the stable run extended to 810, then decayed for 90  (total 900)
  cos-T600+300          continued training the cosine way: from cos-T600's final weights, re-warm for 30 steps
                        and run a new cosine over 300 steps (total 900; new optimizer state)
Every decay ends at 0.1 x peak, the cosine's floor, so the final learning rate is not a second changed variable.
Exact resume makes the segmented stable run identical to a straight one, so a branch continues it exactly.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m07  # noqa: E402

VARIANTS = {   # preset, batch, seq, lr, warmup, scale (multiplies every step count), extra loop args
    "cpu": ("toy", 16, 128, 3e-3, 30, 1, ["--eval-every", "100000", "--eval-windows", "64", "--ckpt-every", "100000",
                                          "--log-every", "1"]),
    "main": ("pilot-30m", 64, 1024, 3e-3, 300, 10, ["--eval-every", "100000", "--ckpt-every", "500", "--log-every", "20",
                                                    "--dtype", "bf16", "--peak", "H100-SXM"]),
}


def plan(scale):
    s = lambda n: n * scale  # noqa: E731
    branches = [("wsd-T300-d10", s(270), s(30), "linear"), ("wsd-T600-d10", s(540), s(60), "linear"),
                ("wsd-T600-d20", s(480), s(120), "linear"), ("wsd-T600-d10-sqrt", s(540), s(60), "1-sqrt"),
                ("wsd-T900-d10", s(810), s(90), "linear")]
    return branches, [("cos-T300", s(300)), ("cos-T600", s(600))], s(300)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("runs/m07/l74"))
    a = ap.parse_args()
    preset, batch, seq, lr, warmup, scale, extra = VARIANTS[a.variant]
    root = a.out / a.variant
    branches, cosines, extend = plan(scale)
    common = ["--preset", preset, "--batch", str(batch), "--seq", str(seq), "--lr", f"{lr:g}", "--warmup", str(warmup),
              "--seed", "0", *extra]
    starts = sorted({b[1] for b in branches})
    stable = root / "stable"
    stable_steps = starts[-1]
    if a.print:
        print("stable:", ["--run", str(stable), "--steps", str(stable_steps), "--schedule", "constant", *common])
    ck = stable                                           # kept copies live next to the run: parent_run = "stable"
    ck.mkdir(parents=True, exist_ok=True)
    for s0 in starts:                                     # train the stable run in segments, keep a copy at each S0
        dst = ck / f"step{s0}.pt"
        if dst.exists() or a.print:
            continue
        m07.optim_train.main(["--run", str(stable), "--steps", str(stable_steps), "--schedule", "constant",
                              "--stop-after", str(s0), *common, "--question", "lesson 07.4: stable (constant) run"])
        shutil.copyfile(stable / "checkpoint.pt", dst)
        print(f"kept stable checkpoint at step {s0}")
    for name, s0, d, shape in branches:
        argv = ["--steps", str(s0 + d), "--schedule", "wsd", "--decay-start", str(s0), "--decay-steps", str(d),
                "--decay-shape", shape, "--min-lr-ratio", "0.1", "--branch-from", str(ck / f"step{s0}.pt"), *common,
                "--question", f"lesson 07.4: decay branch {name}"]
        if a.print:
            print(name, argv)
            continue
        m07.run_arm(root / name, argv, s0 + d)
    for name, total in cosines:
        argv = ["--schedule", "cosine", *common, "--question", f"lesson 07.4: cosine {name}"]
        if a.print:
            print(name, ["--steps", total, *argv])
            continue
        m07.run_arm(root / name, argv, total)
    # continued training the cosine way: new run initialised from cos-T600's final weights, fresh optimizer
    name = "cos-T600+300"
    argv = ["--init-from", str(root / "cos-T600" / "checkpoint.pt"), "--schedule", "cosine", *common,
            "--question", "lesson 07.4: continued training after a finished cosine run"]
    if a.print:
        print(name, ["--steps", extend, *argv])
        return
    m07.run_arm(root / name, argv, extend)


if __name__ == "__main__":
    main()
