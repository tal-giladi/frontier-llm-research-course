"""Run the correctness suite on MLA (frontierlab.attention.mla) and on your absorbed attention.

    python labs/module-03/lesson-01/check_mla.py
    LAB_TARGET=solution python labs/module-03/lesson-01/check_mla.py

Every check must pass before any MLA number (memory, latency, quality) counts.
"""

from pathlib import Path

import torch

from frontierlab.attention import accounting, bench, mla  # noqa: F401  (mla registers "mla")
from frontierlab.attention.checks import attention_suite
from frontierlab.labkit import load_target
from frontierlab.model import LM, toy

lab = load_target(str(Path(__file__).parent / "test_lab.py"))
V = 61
extra = {"kv_lora_rank": 24, "qk_rope_head_dim": 8}
torch.manual_seed(0)
model = LM(toy(vocab_size=V).with_(attention="mla", extra=extra))
tiny = toy(vocab_size=V).with_(attention="mla", hidden_size=8, num_attention_heads=2, head_dim=4,
                               extra={"kv_lora_rank": 6, "qk_rope_head_dim": 2})

print("1. frontierlab MLA, correctness suite (float64)")
attention_suite(model, tiny, V)

print("\n2. your absorbed_attention against the module's naive path (float64)")
mod = LM(toy(vocab_size=V).with_(attention="mla", extra=extra)).double().model.layers[0].self_attn
x = torch.randn(2, 12, 128, dtype=torch.float64)
pos = torch.arange(12)
cos, sin = mod.rope(pos)
q_nope, q_rope = mod.queries(x, cos, sin)
c, k_rope = mod.latents(x, cos, sin)
for T in (12, 1, 4):
    ref = mod.naive(q_nope[:, :, -T:], q_rope[:, :, -T:], c, k_rope, pos[-T:], pos)
    try:
        out = lab.absorbed_attention(q_nope[:, :, -T:], q_rope[:, :, -T:], c, k_rope, mod.kv_b_proj.weight,
                                     mod.H, mod.d_n, mod.d_v, pos[-T:], pos, mod.scale)
        err = (out - ref).abs().max().item()
        print(f"{'PASS' if err < 1e-12 else 'FAIL'}  {T} new token(s) against 12 cached: max |diff| = {err:.2e}")
    except NotImplementedError as e:
        print(f"TODO  {e}")
        break

print("\n3. what the cache holds after 20 tokens (fp32 on CPU)")
cache = bench.prefill(model.eval(), 20)
for name, n in bench.cache_breakdown(cache).items():
    print(f"   {name:7s} {n:7d} bytes")
print(f"   Cache.nbytes() = {cache.nbytes()}  formula (accounting.cache_bytes) = "
      f"{accounting.cache_bytes(model.config, 20, bytes_per=4):.0f}")
