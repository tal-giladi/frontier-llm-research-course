"""Lab 07.2, step 3: attention-logit growth under Muon, and two fixes (QK-Clip, QK-norm).

    python labs/module-07/lesson-02/train_arms.py                  # free CPU (toy preset)
    python labs/module-07/lesson-02/train_arms.py --variant t4     # free GPU (T4, fp32, pilot-10m)
    python labs/module-07/lesson-02/train_arms.py --variant main   # main path (pilot-30m, 1x H100/A100), not run in this build
    python labs/module-07/lesson-02/train_arms.py --variant main70 # pilot ladder rung 2 (pilot-70m), not run in this build

Arms (same preset, data order, seed and token budget; raised learning rate 1e-2, where lesson 03.3 saw logits grow):
  adamw-noqk    AdamW, no QK-norm                     (lesson 03.3's no-qknorm@hi arm, for reference)
  muon-noqk     Muon (match_rms), no QK-norm          logit growth expected (hypothesis H1)
  muon-clip     Muon, no QK-norm, QK-Clip tau = 100   Kimi K2's tau (H2: the clip binds only if logits pass 100)
  muon-clip-lo  Muon, no QK-norm, QK-Clip tau = 15    a tau chosen to bind at course scale (labelled induced)
  muon-qknorm   Muon with QK-norm                     DeepSeek-V4's choice: normalise q and k instead (H3)
  mla-muon      MLA (d_c = K·d, d_r = d/2), Muon      MLA cannot use QK-norm on keys cheaply (Kimi K2's reason)
  mla-clip-lo   MLA, Muon, QK-Clip tau = 15           per-head clip of q^C, k^C, q^R; shared k^R untouched
Every run logs stability.jsonl every step. Runs go to runs/m07/l72/<variant>/<arm>.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m07  # noqa: E402

VARIANTS = {   # preset, steps, batch, seq, extra loop args
    "cpu": ("toy", 300, 16, 128, ["--eval-every", "300", "--ckpt-every", "100", "--eval-windows", "64", "--log-every", "1"]),
    "t4": ("pilot-10m", 2000, 32, 512, ["--dtype", "fp32", "--eval-every", "500", "--ckpt-every", "250", "--log-every", "5"]),
    "main": ("pilot-30m", 4000, 64, 1024, ["--dtype", "bf16", "--eval-every", "500", "--ckpt-every", "500",
                                            "--log-every", "10", "--peak", "H100-SXM"]),
    "main70": ("pilot-70m", 4000, 64, 1024, ["--dtype", "bf16", "--eval-every", "500", "--ckpt-every", "500",
                                              "--log-every", "10", "--peak", "H100-SXM"]),   # second rung of the pilot ladder
}
LR = 1e-2
ARMS = {
    "adamw-noqk": ["--optimizer", "adamw", "--qk-norm", "off"],
    "muon-noqk": ["--optimizer", "muon", "--qk-norm", "off"],
    "muon-clip": ["--optimizer", "muon", "--qk-norm", "off", "--qk-clip", "100"],
    "muon-clip-lo": ["--optimizer", "muon", "--qk-norm", "off", "--qk-clip", "15"],
    "muon-qknorm": ["--optimizer", "muon", "--qk-norm", "on"],
    "mla-muon": ["--optimizer", "muon", "--attention", "mla"],
    "mla-clip-lo": ["--optimizer", "muon", "--attention", "mla", "--qk-clip", "15"],
}


def mla_extra(preset):
    from frontierlab.model import PRESETS
    c = PRESETS[preset]()
    return ["--extra", '{"kv_lora_rank": %d, "qk_rope_head_dim": %d}' % (c.num_key_value_heads * c.head_dim, c.head_dim // 2)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/m07/l72"))
    a = ap.parse_args()
    preset, steps, batch, seq, extra = VARIANTS[a.variant]
    for arm, flags in ARMS.items():
        if a.only and arm not in a.only:
            continue
        argv = ["--preset", preset, "--batch", str(batch), "--seq", str(seq), "--lr", str(LR), "--seed", str(a.seed),
                "--stability-log", "--question", f"lesson 07.2: logit growth under Muon, arm {arm}", *flags, *extra]
        if "mla" in arm:
            argv += mla_extra(preset)
        if a.max_minutes:
            argv += ["--max-minutes", str(a.max_minutes)]
        print(f"\n=== {arm}")
        dt = m07.run_arm(a.out / a.variant / arm, argv, steps)
        if dt:
            print(f"{arm}: {dt:.0f} s")


if __name__ == "__main__":
    main()
