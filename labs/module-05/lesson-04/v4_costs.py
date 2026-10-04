"""Lab 05.4: what CSA/HCA-style compression does to KV memory and per-token attention work.

    python labs/module-05/lesson-04/v4_costs.py                        # checks your lab.py
    python labs/module-05/lesson-04/v4_costs.py --m 4 --m2 128 --topk 512 --window 128

DeepSeek-V4's report does not state m, m', k or n_win in section 2.3, so every number this script prints is
ILLUSTRATIVE: it shows how the costs depend on those hyperparameters, not what V4 costs. The defaults are
round numbers chosen for the illustration, not V4's values. Entry size: one shared-KV MQA entry of width
``--entry-dim`` (also an illustrative value), stored either all in BF16 or as V4 documents it (FP8 except the
last 64 RoPE dimensions in BF16).
"""

import argparse
import os
from pathlib import Path

from frontierlab.attention import compressed
from frontierlab.attention.accounting import human_bytes
from frontierlab.labkit import load_target

lab = load_target(str(Path(__file__).parent / "test_lab.py"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=4, help="CSA compression rate (ILLUSTRATIVE)")
    ap.add_argument("--m2", type=int, default=128, help="HCA compression rate m' (ILLUSTRATIVE)")
    ap.add_argument("--topk", type=int, default=512, help="CSA selected compressed entries (ILLUSTRATIVE)")
    ap.add_argument("--window", type=int, default=128, help="CSA uncompressed recent window (ILLUSTRATIVE)")
    ap.add_argument("--layers", type=int, default=60)
    ap.add_argument("--entry-dim", type=int, default=512, help="shared-KV entry width (ILLUSTRATIVE)")
    ap.add_argument("--rope-dim", type=int, default=64, help="RoPE dims kept in BF16 (documented: last 64)")
    a = ap.parse_args()
    print(f"checking {os.environ.get('LAB_TARGET', 'lab')}.py   ILLUSTRATIVE hyperparameters: m={a.m}, m'={a.m2}, "
          f"k={a.topk}, n_win={a.window}, {a.layers} layers, entry width {a.entry_dim}\n")
    bf16 = 2 * a.entry_dim
    mixed = 1 * (a.entry_dim - a.rope_dim) + 2 * a.rope_dim
    print(f"bytes per entry: all BF16 {bf16} B; FP8 + BF16 RoPE dims {mixed} B ({mixed / bf16:.0%} of BF16)\n")
    layouts = {"all dense": ["dense"] * a.layers, "all CSA": ["csa"] * a.layers, "all HCA": ["hca"] * a.layers,
               "CSA/HCA alternating": ["csa", "hca"] * (a.layers // 2)}
    print(f"{'layout':<22s} {'S':>9s} {'KV (BF16)':>12s} {'KV (FP8 mix)':>13s} {'vs dense':>9s}  "
          f"entries attended per decode query (per layer, mean)")
    for S in (32768, 131072, 1 << 20):
        dense = lab.layout_kv_bytes(S, layouts["all dense"], m=a.m, m2=a.m2, window=a.window, entry_bytes=bf16)
        for name, layers in layouts.items():
            kv = lab.layout_kv_bytes(S, layers, m=a.m, m2=a.m2, window=a.window, entry_bytes=bf16)
            kv8 = lab.layout_kv_bytes(S, layers, m=a.m, m2=a.m2, window=a.window, entry_bytes=mixed)
            att = sum(compressed.attended_entries(S, kind=k, m=a.m if k == "csa" else a.m2, k=a.topk,
                                                  window=a.window) for k in layers) / len(layers)
            print(f"{name:<22s} {S:>9d} {human_bytes(kv):>12s} {human_bytes(kv8):>13s} {kv / dense:9.3f}  {att:,.0f}")
        print()
    S = 1 << 20
    print(f"CSA indexer work per decode query at S = {S}: {S // a.m:,} compressed keys scored "
          f"(vs {S:,} for DSA over raw tokens): the O(L^2) term shrinks by m = {a.m}, it does not disappear.")
    print("Causality: with m = 4 a query at position 10 may use", lab.usable_entries(10, a.m),
          "compressed entries (tokens 0-7); tokens 8-10 must come from the sliding window.")


if __name__ == "__main__":
    main()
