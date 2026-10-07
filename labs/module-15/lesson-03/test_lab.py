import math

from frontierlab.labkit import load_target
from frontierlab.perf.roofline import HARDWARE
from frontierlab.ttc import serving as SV

lab = load_target(__file__)


def test_kv_per_token_released_shapes():
    # Qwen3-1.7B: 28 layers, 8 KV heads of 128 -> 114,688 bytes per token in bf16
    assert lab.kv_bytes_per_token(28, "gqa", kv_heads=8, head_dim=128) == 114688
    # DeepSeek-V3: 61 layers, latent 512 + RoPE key 64 -> 70,272 bytes per token in bf16
    assert lab.kv_bytes_per_token(61, "mla", d_c=512, d_r=64) == 70272
    assert lab.kv_bytes_per_token(28, "gqa", kv_heads=8, head_dim=128, bytes_per=1) == 57344   # FP8 cache


def test_against_course_accounting():
    d = SV.stage_d_designs()
    S = 131072
    assert lab.kv_bytes_per_token(28, "gqa", kv_heads=8, head_dim=128) * S == SV.kv_bytes_seq(d["GQA (as released)"], S)
    assert lab.kv_bytes_per_token(28, "mla", d_c=512, d_r=64) * S == SV.kv_bytes_seq(d["MLA d_c=4d"], S)
    lg = d["local/global 5:1, w=1024"]
    n_g = sum(1 for i in range(28) if (i + 1) % 6 == 0)
    assert lab.local_global_bytes(S, 28 - n_g, n_g, 1024, 8, 128) == SV.kv_bytes_seq(lg, S)
    assert lab.local_global_bytes(500, 4, 1, 1024, 2, 64) == 5 * 500 * 2 * 2 * 64 * 2


def test_decode_step_and_capacity():
    hw = HARDWARE["H100-SXM"]
    t = lab.decode_step_time(3.4e9, 1.5e9, 8, 4e9, hw.peak_flops, hw.mem_bw)
    assert math.isclose(t, (3.4e9 + 8 * 1.5e9) / hw.mem_bw)
    t_c = lab.decode_step_time(1e6, 0, 10_000, 1e9, hw.peak_flops, hw.mem_bw)
    assert math.isclose(t_c, 10_000 * 1e9 / hw.peak_flops)
    G = 2 ** 30
    assert lab.capacity(80 * G, 3.4e9, 14 * G) == int((72 * G - 3.4e9) // (14 * G))
    assert lab.capacity(8 * G, 9 * G, 1.0) == 0
