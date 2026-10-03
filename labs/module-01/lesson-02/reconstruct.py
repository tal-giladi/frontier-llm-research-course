"""Reconstruct an architecture from config.json and say where every number came from.

    python labs/module-01/lesson-02/reconstruct.py deepseek-ai/DeepSeek-V3            # bundled snapshot
    python labs/module-01/lesson-02/reconstruct.py openai/gpt-oss-20b --fetch          # current Hub file
    python labs/module-01/lesson-02/reconstruct.py my_config.json --seq 8192 --context 131072
    LAB_TARGET=lab python labs/module-01/lesson-02/reconstruct.py Qwen/Qwen3-8B        # also check your lab.py

The first table maps each ArchSpec field to the config key it was read from. Fields marked
"model_type (...)" are not in the file at all: they come from the modelling code of that family, which
is exactly the kind of fact you must cite from code, not from the config.
"""

import argparse
import os
from pathlib import Path

from frontierlab.calc import (decode_flops_per_token, fetch_config, flops_per_token, from_hf, kv_bytes,
                              param_counts)

FIELDS = ("vocab_size", "hidden_size", "num_layers", "layer_kinds", "num_heads", "num_kv_heads", "head_dim",
          "sliding_window", "kv_lora_rank", "qk_rope_dim", "n_experts", "top_k", "expert_intermediate",
          "n_shared", "dense_layers", "qk_norm", "norms_per_layer", "tie_embeddings", "max_context", "mtp_layers")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--context", type=int, default=32768)
    a = ap.parse_args()
    s = from_hf(fetch_config(a.config) if a.fetch else a.config, name=a.config)
    print(f"{'field':22s} {'value':28s} config key")
    for f in FIELDS:
        v = getattr(s, f)
        if isinstance(v, list) and len(v) > 6:
            v = {k: v.count(k) for k in dict.fromkeys(v)} if isinstance(v[0], str) else f"{v[:3]}..{v[-1]} ({len(v)})"
        if f in s.source or v not in (0, None, [], False, "none"):
            print(f"{f:22s} {str(v):28s} {s.source.get(f, '-')}")
    pc = param_counts(s)
    print(f"\ntotal {pc['total']:,}  active {pc['active']:,} (head counted, input embedding not)")
    print(f"training FLOPs/token at T={a.seq}: {flops_per_token(s, a.seq):.4g}   6*active = {6 * pc['active']:.4g}")
    print(f"decode FLOPs/token at S={a.context}: {decode_flops_per_token(s, a.context):.4g}")
    print(f"KV cache at S={a.context}: {kv_bytes(s, a.context) / 2**30:.3f} GiB (BF16), "
          f"{kv_bytes(s, a.context, 1) / 2**30:.3f} GiB (FP8)")
    if "LAB_TARGET" in os.environ:
        from frontierlab.labkit import load_target
        lab = load_target(str(Path(__file__).parent / "test_lab.py"))
        print(f"\nyour lab.py: active {lab.active_params(s):,}  kv elements {lab.kv_elements(s, a.context):,}")


if __name__ == "__main__":
    main()
