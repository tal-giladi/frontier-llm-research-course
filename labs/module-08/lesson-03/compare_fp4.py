"""Lab 08.3, step 3b: the FP4 arms against BF16 and the contract's rule for each ingredient.

    python labs/module-08/lesson-03/compare_fp4.py                  # runs/m08/l83/cpu
    python labs/module-08/lesson-03/compare_fp4.py --variant t4 --device cuda

Held-out loss on 256 fixed windows of 128 tokens, each arm evaluated in the precision it trained with; gaps to
BF16 seed 0, paired by window (95% bootstrap). Noise: the BF16 and NVFP4 seed-to-seed differences. An ingredient
"matters at this scale" if removing it widens the gap to BF16 by more than both the NVFP4 seed difference and the
half-width of the paired interval of (ablation - nvfp4).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m08  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="cpu")
    ap.add_argument("--root", type=Path, default=Path("runs/m08/l83"))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    root = a.root / a.variant
    L = {p.name: m08.eval_losses(p, device=a.device) for p in sorted(root.iterdir()) if (p / "checkpoint.pt").exists()}
    ref = L["bf16-s0"]
    print(f"{'arm':18s} {'held-out':>9s} {'gap vs bf16-s0 [95% CI]':>32s} {'train tok/s':>11s}")
    for name, l in L.items():
        tr = m08.train_rows(root / name)
        tps = sorted(r["tok_per_s"] for r in tr if r["step"] > 20)
        c = "" if name == "bf16-s0" else m08.fmt(m08.compare(l, ref))
        print(f"{name:18s} {m08.mean(l):9.4f} {c:>32s} {tps[len(tps) // 2] if tps else float('nan'):11.0f}")
    noise = {k: abs(m08.mean(L[f"{k}-s1"]) - m08.mean(L[f"{k}-s0"])) for k in ("bf16", "nvfp4") if f"{k}-s1" in L}
    print("\nseed-to-seed |difference| of the held-out mean:", {k: round(v, 4) for k, v in noise.items()})
    base_gap = m08.mean(L["nvfp4-s0"]) - m08.mean(ref)
    for abl, what in (("nvfp4-no-sr", "stochastic rounding"), ("nvfp4-no-rht", "the Hadamard transform"),
                      ("nvfp4-keep-s0", "BF16 last block (added, not removed)"), ("mxfp4-s0", "NVFP4 scaling vs naive MXFP4")):
        key = abl if abl.endswith("-s0") else f"{abl}-s0"
        if key not in L:
            continue
        c = m08.compare(L[key], L["nvfp4-s0"])
        half = (c["hi"] - c["lo"]) / 2
        widened = c["diff"]
        verdict = ("matters at this scale" if abs(widened) > max(noise.get("nvfp4", 0.0), half) else "no detectable effect")
        print(f"{key:16s} vs nvfp4-s0: {m08.fmt(c)}  ({what}); threshold {max(noise.get('nvfp4', 0.0), half):.4f} -> {verdict}")
    print(f"\nnvfp4 gap to bf16 (seed 0): {base_gap:+.4f} nats = {100 * base_gap / m08.mean(ref):+.2f}% relative "
          "(the NVFP4 paper reports <1% relative to FP8 at 12B / 10T tokens; a different baseline, model and scale)")
    log = m08.precision_rows(root / "nvfp4-s0")
    if log:
        r = log[-1]
        print(f"precision log nvfp4-s0 step {r['step']}: grad rel err {r.get('grad_rel_err', 0):.3f}, "
              f"grad underflow {r.get('grad_underflow', 0):.3f}, act underflow {r.get('act_underflow', 0):.3f}")


if __name__ == "__main__":
    main()
