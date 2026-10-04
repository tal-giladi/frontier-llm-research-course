"""Lab 05.1: recurrent vs chunked-parallel equivalence of your delta rule, and state vs KV memory.

    python labs/module-05/lesson-01/equivalence.py                  # checks your lab.py
    LAB_TARGET=solution python labs/module-05/lesson-01/equivalence.py

Part 1 — the same function computed two ways must agree: float64 to rounding (~1e-15), float32 to
~1e-6. A chunk-size dependence beyond that is a bug, not "numerical noise".
Part 2 — what a linear layer's state costs against a full layer's KV cache, from the configs of
Baseline-0's 3:1 hybrid, Qwen3-Next-80B-A3B and Kimi Linear 48B-A3B.
"""

import os
import time
from pathlib import Path

import torch

from frontierlab.attention import accounting, deltanet
from frontierlab.attention.accounting import human_bytes
from frontierlab.labkit import load_target
from frontierlab.model import baseline0

lab = load_target(str(Path(__file__).parent / "test_lab.py"))
print(f"checking {os.environ.get('LAB_TARGET', 'lab')}.py\n")

print("Part 1. Recurrent vs chunked (B=2, H=4, d_k=d_v=32, random decays in (exp(-2), 1))")
print(f"{'dtype':8s} {'T':>5s} {'chunk':>5s} {'max |o_rec - o_chunk|':>22s} {'max |S_rec - S_chunk|':>22s}  time rec / chunk")
for dtype in (torch.float64, torch.float32):
    for T, chunk in ((64, 1), (64, 16), (256, 16), (256, 64), (1024, 64)):
        g = torch.Generator().manual_seed(T + chunk)
        f = lambda *s: torch.randn(*s, generator=g).to(dtype)  # noqa: E731
        q, k, v = f(2, 4, T, 32) * 32 ** -0.5, deltanet.l2norm(f(2, 4, T, 32)), f(2, 4, T, 32)
        gate = (-torch.rand(2, 4, T, 32, generator=g) * 2).to(dtype)
        beta = torch.rand(2, 4, T, generator=g).to(dtype)
        t0 = time.perf_counter()
        o1, S1 = lab.delta_recurrent(q, k, v, gate, beta)
        t1 = time.perf_counter()
        o2, S2 = lab.delta_chunked(q, k, v, gate, beta, None, chunk)
        t2 = time.perf_counter()
        print(f"{str(dtype)[6:]:8s} {T:5d} {chunk:5d} {(o1 - o2).abs().max().item():22.2e} "
              f"{(S1 - S2).abs().max().item():22.2e}  {t1 - t0:6.3f} s / {t2 - t1:6.3f} s")

print("\nPart 2. Decode memory per sequence (K/V in BF16, linear state in fp32 unless stated)")
b0 = baseline0()
hyb = b0.with_(attention="hybrid", extra={"full_every": 4})
pattern = lab.hybrid_pattern(12, 4)
kv = 2 * b0.num_key_value_heads * b0.head_dim * 2
st = accounting.linear_state_bytes(hyb, bytes_per=2, state_bytes=4)
print(f"Baseline-0 3:1 hybrid layout {''.join('F' if p == 'full' else 'L' for p in pattern)}; "
      f"one linear layer's state: {human_bytes(st)} (12 heads x 64 x 64 x 4 B + conv tail)")
for S in (4096, 32768, 131072, 1 << 20):
    dense = accounting.m05_cache_bytes(b0, S)
    h = lab.cache_bytes(pattern, S, kv, st)
    print(f"  S={S:>8d}  Baseline-0 {human_bytes(dense):>11s}   hybrid {human_bytes(h):>11s}   ratio {h / dense:.3f}")

# Qwen3-Next-80B-A3B (config.json snapshot): 48 layers, full_attention_interval 4; full: 2 KV heads x 256;
# linear: 32 value heads with d_k = d_v = 128 (grouped: 16 key heads shared by 32 value heads).
qn_full = 2 * 2 * 256 * 2
qn_state = 32 * 128 * 128
print("\nQwen3-Next-80B-A3B (12 full + 36 Gated DeltaNet layers):")
for sb, label in ((2, "BF16 state"), (4, "fp32 state")):
    for S in (32768, 262144):
        tot = 12 * S * qn_full + 36 * qn_state * sb
        print(f"  {label}  S={S:>7d}  {human_bytes(tot):>11s}  ({tot / S / 1024:.1f} KiB/token; states {human_bytes(36 * qn_state * sb)})")
# Kimi Linear 48B-A3B (config.json): 27 layers, 7 MLA (latent 512 + rope 64) and 20 KDA (32 heads x 128 x 128).
kl_full = (512 + 64) * 2
kl_state = 32 * 128 * 128
print("Kimi Linear 48B-A3B (7 MLA + 20 KDA layers), BF16 latent, fp32 state:")
for S in (131072, 1 << 20):
    tot = 7 * S * kl_full + 20 * kl_state * 4
    all_mla = 27 * S * kl_full
    print(f"  S={S:>8d}  {human_bytes(tot):>11s} vs all-MLA {human_bytes(all_mla):>11s}  reduction {1 - tot / all_mla:.1%}")
