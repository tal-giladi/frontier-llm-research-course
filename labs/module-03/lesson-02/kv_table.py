"""Lab 03.2: KV-cache bytes per token for MHA / GQA / MQA / MLA / windowed mixes at 32K-1M, by formula,
checked against real caches.

    python labs/module-03/lesson-02/kv_table.py

Part 1 uses Baseline-0's shape (12 layers, 12 query heads of width 64) and changes only the attention.
Part 2 reads the bundled config snapshots of real models (frontierlab.calc). Part 3 builds toy models
of each kind, decodes 300 tokens and compares Cache.nbytes() (minus the int64 position bookkeeping) with
the formula. Contexts beyond a model's trained length are arithmetic only, not a claim that it works there.
"""

from pathlib import Path

import torch

from frontierlab.attention import accounting, bench, gated, mla, sliding  # noqa: F401  (registers kinds)
from frontierlab.calc import from_hf, kv_bytes as calc_kv_bytes
from frontierlab.labkit import load_target
from frontierlab.model import LM, baseline0, toy

lab = load_target(str(Path(__file__).parent / "test_lab.py"))
CONTEXTS = (32768, 131072, 1048576)


def row(name, per_token_at):
    cells = "".join(f"{per_token_at(S) / 1024:>11.2f}" for S in CONTEXTS)
    gib = per_token_at(CONTEXTS[1]) * CONTEXTS[1] / 2**30
    return f"  {name:<34s}{cells}{gib:>12.2f}"


b0 = baseline0()
designs = {
    "MHA (K = 12)": b0.with_(num_key_value_heads=12),
    "GQA K = 4 (Baseline-0)": b0,
    "GQA K = 2": b0.with_(num_key_value_heads=2),
    "MQA (K = 1)": b0.with_(num_key_value_heads=1),
    "MLA d_c = 256, d_r = 32": b0.with_(attention="mla", extra={"kv_lora_rank": 256, "qk_rope_head_dim": 32}),
    "all sliding, w = 1024": b0.with_(attention="sliding", extra={"window": 1024}),
    "5 local : 1 global, w = 1024": b0.with_(attention="local_global", extra={"global_every": 6, "window": 1024}),
    "1 local : 1 global, w = 128": b0.with_(attention="local_global", extra={"global_every": 2, "window": 128}),
}
print("1. Baseline-0 shape, BF16, KiB per token (average over the context) and GiB per sequence at 128K")
print(f"  {'design':<34s}" + "".join(f"{f'S={S // 1024}K':>11s}" for S in CONTEXTS) + f"{'GiB@128K':>12s}")
for name, cfg in designs.items():
    print(row(name, lambda S, c=cfg: accounting.kv_bytes(c, S) / S))

print("\n2. Real configs (bundled snapshots, frontierlab.calc), BF16")
print(f"  {'model':<34s}" + "".join(f"{f'S={S // 1024}K':>11s}" for S in CONTEXTS) + f"{'GiB@128K':>12s}")
for name in ("Qwen/Qwen3-8B", "unsloth/gemma-3-27b-pt", "openai/gpt-oss-120b", "deepseek-ai/DeepSeek-V3",
             "moonshotai/Kimi-K2-Instruct"):
    spec = from_hf(name)
    print(row(name.split("/")[1], lambda S, s=spec: calc_kv_bytes(s, S) / S))

print("\n3. Measured: toy models, 300 tokens decoded (prefill 100, then 25-token chunks), fp32")
kinds = {"gqa": {}, "mla": {"kv_lora_rank": 64, "qk_rope_head_dim": 16}, "sliding": {"window": 32},
         "local_global": {"window": 32, "global_every": 4}}
for kind, extra in kinds.items():
    cfg = toy().with_(attention=kind, extra=extra)
    m = LM(cfg).eval()
    cache = bench.prefill(m, 100)
    with torch.no_grad():
        for _ in range(8):
            m(torch.zeros(1, 25, dtype=torch.long), cache=cache)
    pos = bench.cache_breakdown(cache).get("pos", 0)
    formula = accounting.kv_bytes(cfg, cache.length, bytes_per=4)
    print(f"  {kind:<14s} tokens {cache.length}  measured K/V or latent {cache.nbytes() - pos:>8,d} B   "
          f"formula {formula:>10,.0f} B   {'OK' if cache.nbytes() - pos == formula else 'MISMATCH'}")

try:
    pat = lab.layer_pattern(12, 6)
    mine = lab.kv_bytes(pat, 131072, 2 * 4 * 64 * 2, 1024)          # K and V x 4 KV heads x 64 x BF16
    ref = accounting.kv_bytes(designs["5 local : 1 global, w = 1024"], 131072)
    print(f"\n4. your kv_bytes for 5:1, w = 1024 at 128K: {mine / 2**30:.3f} GiB (frontierlab: {ref / 2**30:.3f} GiB)")
except NotImplementedError as e:
    print(f"\n4. {e}")
