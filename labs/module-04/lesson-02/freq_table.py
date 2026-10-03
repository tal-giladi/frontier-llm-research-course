"""Per-frequency table of the RoPE rules: what each rule does to each channel pair.

    python labs/module-04/lesson-02/freq_table.py                                  # toy: D=32, 256 -> 1024
    python labs/module-04/lesson-02/freq_table.py --dim 64 --train-len 1024 --factor 32   # Baseline-0, 1K -> 32K

Columns: pair i, wavelength lambda_i = 2 pi / theta_i in tokens, rotations r_i = L / lambda_i over the
trained length, and for each rule the divisor theta_i / theta_i' (1 = unchanged, s = fully
interpolated). The YaRN columns show both ramps: the code's (linear in i, rounded outwards, which
Transformers and released checkpoints use) and the paper's (linear in r).
"""

import argparse

from frontierlab.longctx.rope import frequency_table, yarn_attention_factor, yarn_correction_dim


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dim", type=int, default=32, help="rotated channels per head")
    ap.add_argument("--base", type=float, default=1e4)
    ap.add_argument("--train-len", type=int, default=256)
    ap.add_argument("--factor", type=float, default=4.0)
    ap.add_argument("--beta-fast", type=float, default=32.0)
    ap.add_argument("--beta-slow", type=float, default=1.0)
    a = ap.parse_args()
    rows = frequency_table(a.dim, a.base, a.factor, a.train_len, a.beta_fast, a.beta_slow)
    lo = yarn_correction_dim(a.beta_fast, a.dim, a.base, a.train_len)
    hi = yarn_correction_dim(a.beta_slow, a.dim, a.base, a.train_len)
    print(f"D={a.dim} base={a.base:g} L={a.train_len} s={a.factor:g}: YaRN keeps pairs with more than {a.beta_fast:g} "
          f"rotations (i < {lo:.2f}), interpolates pairs with fewer than {a.beta_slow:g} (i > {hi:.2f}); "
          f"attention factor {yarn_attention_factor(a.factor):.4f} (logits x {yarn_attention_factor(a.factor) ** 2:.4f})\n")
    print(f"{'i':>3} {'wavelength':>12} {'rotations':>10} {'PI':>7} {'NTK':>7} {'YaRN code':>10} {'YaRN paper':>11}")
    for r in rows:
        print(f"{r['pair']:>3} {r['wavelength']:>12.1f} {r['rotations']:>10.3f} {r['pi']:>7.3f} {r['ntk']:>7.3f} "
              f"{r['yarn_code']:>10.3f} {r['yarn_paper']:>11.3f}")


if __name__ == "__main__":
    main()
