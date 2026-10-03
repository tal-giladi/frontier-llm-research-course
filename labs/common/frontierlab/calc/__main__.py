"""Command line: print the calculator's view of one or more configs.

    python -m frontierlab.calc Qwen/Qwen3-8B                  # a bundled snapshot
    python -m frontierlab.calc path/to/config.json --seq 4096 --context 131072
    python -m frontierlab.calc openai/gpt-oss-20b --fetch     # download the current config from the Hub
    python -m frontierlab.calc --all                          # every snapshot, one table
"""

import argparse

from frontierlab.calc.arith import (decode_flops_per_token, flops_per_token, kv_bytes, kv_bytes_per_token,
                                    param_counts)
from frontierlab.calc.hfconfig import fetch_config, from_hf, snapshot_names


def describe(spec, T: int, S: int) -> str:
    pc = param_counts(spec)
    kinds = ", ".join(f"{spec.layer_kinds.count(k)} {k}" for k in ("full", "sliding", "linear")
                      if k in spec.layer_kinds)
    lines = [f"{spec.name}  ({spec.model_type}, {spec.attention}; {spec.num_layers} layers: {kinds})",
             f"  heads {spec.num_heads}, kv heads {spec.num_kv_heads}, head dim {spec.head_dim}"
             + (f", window {spec.sliding_window}" if spec.sliding_window else ""),
             f"  params total {pc['total'] / 1e9:.3f}B   active {pc['active'] / 1e9:.3f}B"
             f"   (embedding {pc['embedding'] / 1e9:.3f}B, tied={spec.tie_embeddings})",
             f"  training FLOPs/token at T={T}: {flops_per_token(spec, T) / 1e9:.2f} GFLOP"
             f"   (6*active = {6 * pc['active'] / 1e9:.2f})",
             f"  decode FLOPs/token at S={S}: {decode_flops_per_token(spec, S) / 1e9:.2f} GFLOP",
             f"  KV cache at S={S} (BF16): {kv_bytes(spec, S) / 2**30:.2f} GiB per sequence,"
             f" {kv_bytes_per_token(spec, S) / 1024:.1f} KiB/token"]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fetch", action="store_true", help="download config.json from the Hub instead of a snapshot")
    ap.add_argument("--seq", type=int, default=4096, help="training sequence length T")
    ap.add_argument("--context", type=int, default=32768, help="decode context S")
    a = ap.parse_args(argv)
    names = snapshot_names() if a.all else a.configs
    for n in names:
        spec = from_hf(fetch_config(n) if a.fetch else n, name=n)
        print(describe(spec, a.seq, a.context))


if __name__ == "__main__":
    main()
