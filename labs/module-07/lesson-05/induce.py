"""Lab 07.5, step 2: induce three kinds of failure, and try the published fixes on the first.

    python labs/module-07/lesson-05/induce.py                    # free CPU (toy preset), about 50 minutes
    python labs/module-07/lesson-05/induce.py --variant main     # main path (pilot-30m), not run in this build
    python labs/module-07/lesson-05/induce.py --only healthy f3-data

Every run: toy preset, MuonAdamW in its AdamW configuration (so QK-Clip and the stability log work), 300 steps of
16 x 128 tokens, seed 0, the loop's cosine schedule with 30 warmup steps, gradient clipping at 1.0, the loss and
gradient norm logged every step, stability.jsonl every step.

  healthy       QK-norm on, lr 3e-3                       the reference; its checkpoint at step 150 is kept
  f1-logits     QK-norm off, lr 1e-2                      attention-logit growth (lessons 03.3, 07.2)
  f2-lr-bug     healthy's step-150 checkpoint resumed with --lr 3e-2 (a typo on restart)   optimizer failure
  f3-data       healthy, but steps 150-152 train on windows of one repeated 4-token pattern   data failure
  fixes for f1 (each changes one thing from f1-logits):
  f1+qknorm     QK-norm back on                           (Baseline-0, Gemma 3, OLMo 2, DeepSeek-V4's choice)
  f1+softcap    attention-logit soft-cap 50               (Gemma 2's attention cap)
  f1+clipqkv    q, k, v projections clamped to [-8, 8]    (OLMo-1.7-7B's clip_qkv)
  f1+qkclip     QK-Clip with tau 15                       (Kimi K2's mechanism; tau scaled to this model, lesson 07.2)
  f1+zloss      z-loss 1e-4                               (PaLM's z-loss: a control, it targets the *output* softmax)
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m07  # noqa: E402

VARIANTS = {  # preset, steps, batch, seq, extra
    "cpu": ("toy", 300, 16, 128, ["--eval-every", "100000", "--eval-windows", "64", "--ckpt-every", "150",
                                  "--log-every", "1", "--warmup", "30"]),
    "main": ("pilot-30m", 4000, 64, 1024, ["--eval-every", "100000", "--ckpt-every", "500", "--log-every", "1",
                                           "--warmup", "300", "--dtype", "bf16", "--peak", "H100-SXM"]),
}

F1 = ["--qk-norm", "off", "--lr", "1e-2"]
ARMS = {
    "f1-logits": F1,
    "f3-data": ["--lr", "3e-3", "--inject-bad-steps", "BAD"],
    "f1+qknorm": ["--qk-norm", "on", "--lr", "1e-2"],
    "f1+softcap": [*F1, "--attn-softcap", "50"],
    "f1+clipqkv": [*F1, "--clip-qkv", "8"],
    "f1+qkclip": [*F1, "--qk-clip", "15"],
    "f1+zloss": [*F1, "--z-loss", "1e-4"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/m07/l75"))
    a = ap.parse_args()
    preset, steps, batch, seq, extra = VARIANTS[a.variant]
    half = steps // 2
    bad = ",".join(str(half + i) for i in range(3))
    root = a.out / a.variant
    common = ["--optimizer", "adamw", "--preset", preset, "--batch", str(batch), "--seq", str(seq), "--seed", "0",
              "--stability-log", *extra]
    want = lambda n: a.only is None or n in a.only  # noqa: E731

    if want("healthy") or want("f2-lr-bug"):
        healthy = root / "healthy"
        keep = healthy / ("step%d.pt" % half)                # next to the run, so parent_run = "healthy"
        hargv = [*common, "--lr", "3e-3", "--question", "lesson 07.5: healthy reference"]
        if not keep.exists():
            m07.optim_train.main(["--run", str(healthy), "--steps", str(steps), "--stop-after", str(half), *hargv])
            shutil.copyfile(healthy / "checkpoint.pt", keep)
        m07.run_arm(healthy, hargv, steps)
        if want("f2-lr-bug"):
            m07.run_arm(root / "f2-lr-bug", [*common, "--lr", "3e-2", "--branch-from", str(keep),
                                             "--question", "lesson 07.5: resumed with the wrong learning rate"], steps)
    for arm, flags in ARMS.items():
        if not want(arm):
            continue
        flags = [f.replace("BAD", bad) for f in flags]
        print(f"\n=== {arm}")
        m07.run_arm(root / arm, [*common, *flags, "--question", f"lesson 07.5: {arm}"], steps)


if __name__ == "__main__":
    main()
