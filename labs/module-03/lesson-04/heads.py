"""Lab 03.4 (extension): what head count and head width cost, and a small training comparison.

    python labs/module-03/lesson-04/heads.py                  # arithmetic only, seconds
    python labs/module-03/lesson-04/heads.py --train          # + 3 toy runs on Data-v0 (free CPU)

1. Kimi K2 (bundled config) with 64 heads, as shipped, and with 128 heads, as DeepSeek-V3: decode
   FLOPs per token at 4K, 32K and 128K, and average prefill FLOPs per token at the same lengths.
   K2 section 2.3 reports an 83% increase in inference FLOPs at 128K for 64 -> 128 heads; our
   arithmetic uses the course convention (naive MLA dims, 2 FLOPs per multiply-add) and does not
   know which convention the report used.
2. Qwen3-Next's attention layers: head_dim 256 with partial_rotary_factor 0.25 -> 64 rotated channels;
   the RoPE wavelengths those channels cover with rope_theta from the config.
3. With --train: b0, wide-heads (H/2 heads of width 2d) and partial-rope (wide-heads + RoPE on 25% of
   channels) for 300 toy steps, seed 0, then held-out loss with a paired interval over 256 windows.
"""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m03  # noqa: E402
from frontierlab.calc import decode_flops_per_token, flops_per_token, from_hf, load_config  # noqa: E402
from frontierlab.labkit import load_target  # noqa: E402

lab = load_target(str(Path(__file__).parent / "test_lab.py"))


def part1():
    k2 = from_hf("moonshotai/Kimi-K2-Instruct")
    k2_128 = copy.deepcopy(k2)
    k2_128.num_heads = 128
    print("1. Kimi K2, 64 vs 128 attention heads (everything else as in the config), GFLOP per token")
    print(f"   {'context':>8s} {'decode 64':>10s} {'decode 128':>11s} {'increase':>9s}   "
          f"{'prefill 64':>10s} {'prefill 128':>12s} {'increase':>9s}")
    for S in (4096, 32768, 131072):
        d64, d128 = decode_flops_per_token(k2, S), decode_flops_per_token(k2_128, S)
        p64, p128 = flops_per_token(k2, S, training=False), flops_per_token(k2_128, S, training=False)
        print(f"   {S:>8d} {d64 / 1e9:10.1f} {d128 / 1e9:11.1f} {d128 / d64 - 1:9.1%}   "
              f"{p64 / 1e9:10.1f} {p128 / 1e9:12.1f} {p128 / p64 - 1:9.1%}")
    try:
        a = lab.attention_flops_per_token(64, 192, 128, 131072) * k2.num_layers
        print(f"   your attention_flops_per_token x 61 layers at 128K, 64 heads: {a / 1e9:.1f} GFLOP "
              f"(attention part of the decode figure above)")
    except NotImplementedError as e:
        print(f"   {e}")


def part2():
    d = load_config("Qwen/Qwen3-Next-80B-A3B-Instruct")
    hd, frac, theta = d["head_dim"], d["partial_rotary_factor"], float(d["rope_theta"])
    r = int(hd * frac)
    print(f"\n2. Qwen3-Next: head_dim {hd}, partial_rotary_factor {frac} -> {r} rotated channels "
          f"({r // 2} frequencies), rope_theta {theta:g}")
    waves = [2 * math.pi * theta ** (2 * i / r) for i in range(r // 2)]
    print(f"   wavelengths in tokens: shortest {waves[0]:.1f}, median {waves[len(waves) // 2]:,.0f}, "
          f"longest {waves[-1]:,.0f}; {hd - r} channels carry no position signal")


def part3(out: Path):
    import train_variant
    from frontierlab.data.loader import TokenData
    from frontierlab.evals.heldout import window_losses
    from frontierlab.stats import paired_bootstrap
    losses = {}
    for arm in ("b0", "wide-heads", "partial-rope"):
        run = out / f"{arm}-s0"
        argv = ["--run", str(run), "--preset", "toy", "--steps", "300", "--batch", "16", "--seq", "128",
                "--eval-every", "300", "--ckpt-every", "100", "--eval-windows", "64",
                "--question", f"lesson 03.4: head shape, arm {arm}"]
        if not (run / "metrics.jsonl").exists() or '"step": 300,' not in (run / "metrics.jsonl").read_text():
            cfg, argv = train_variant.plan(arm, argv)
            print(f"\n=== {arm}: {m03.describe(cfg, 128)}")
            train_variant.train(cfg, argv)
        model, _ = m03.load_run(run)
        losses[arm] = window_losses(model, TokenData("val"), 256, 128)
    print("\n3. held-out loss (256 fixed windows, T = 128), one seed: the interval is evaluation noise only")
    for arm, ls in losses.items():
        line = f"   {arm:<13s} {sum(ls) / len(ls):.4f}"
        if arm != "b0":
            pb = paired_bootstrap(ls, losses["b0"])
            line += f"   vs b0 {pb['mean_diff']:+.4f}  95% CI [{pb['ci'][0]:+.4f}, {pb['ci'][1]:+.4f}]"
        print(line)
    (out / "losses.json").write_text(json.dumps({k: sum(v) / len(v) for k, v in losses.items()}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("runs/m03/l34"))
    a = ap.parse_args()
    part1()
    part2()
    if a.train:
        part3(a.out)
