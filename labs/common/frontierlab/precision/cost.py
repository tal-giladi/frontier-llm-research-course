"""Projected speed-up of low-precision linears, from the Module 2 roofline (lessons 08.2, 08.3 and the project).

Nothing here is a measurement. The model:

* Every op of the training step is timed at its roofline bound (``frontierlab.perf.roofline.forward_ops``,
  backward = 2 × forward), on the BF16 peak.
* The block linears (q, k, v, o, gate, up, down) run at the low-precision peak instead: FP8 = 2 × BF16 on H100
  and L4 (dense 1,979 vs 989 and 242.5 vs 121 TFLOP/s); FP4 = 4 × BF16 on B200 (9 vs 2.25 PFLOP/s per GPU, our
  division of NVIDIA's 8-GPU DGX B200 figures). The output head and attention stay in BF16 (DeepSeek-V3 section
  3.3.1 keeps the output head and attention operators in higher precision).
* The low-precision GEMMs also read smaller operands (1 byte per FP8 value, 0.5 per FP4 value; the output stays
  BF16), which matters for the small, memory-bound projections (k_proj and v_proj with few KV heads).
* Quantisation casts are added as memory traffic unless ``fused_casts``: per linear and step, read BF16 and
  write the low-precision copy of x and W (forward), dy and W again (Dgrad), dy and x again along the other axis
  (Wgrad): 3 bytes per element (2 read + 1 written; FP4 writes 0.5) for each of 2·(M·K) + 2·(M·N) + 2·(N·K).

The result is an upper bound on the gain, because real kernels reach a fraction of either peak and small GEMMs
reach less of the FP8 peak than large ones.
"""

from __future__ import annotations

from dataclasses import replace

from frontierlab.model.config import ModelConfig
from frontierlab.perf.roofline import HARDWARE, Hardware, forward_ops

LOWP_FACTOR = {"fp8": 2.0, "fp4": 4.0}
LINEAR_OPS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
B200 = Hardware("B200 (per GPU, dense BF16 = DGX B200 / 8)", 2.25e15, 8.0e12, "bf16",
                "https://www.nvidia.com/en-us/data-center/dgx-b200/")


def get_hw(name: str) -> Hardware:
    return B200 if name == "B200" else HARDWARE[name]


def projected_speedup(cfg: ModelConfig, B: int, T: int, hw: str | Hardware = "H100-SXM", lowp: str = "fp8",
                      fused_casts: bool = False) -> dict:
    """Roofline step times (s) in BF16 and with low-precision block linears, and their ratio (PROJECTED)."""
    hw = get_hw(hw) if isinstance(hw, str) else hw
    lp = replace(hw, peak_flops=hw.peak_flops * LOWP_FACTOR[lowp])
    t_bf16 = t_lp = t_lin_bf16 = t_lin_lp = 0.0
    M = B * T
    C, I, H, KV, d = cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    shapes = {"q_proj": (C, H * d), "k_proj": (C, KV * d), "v_proj": (C, KV * d), "o_proj": (H * d, C),
              "gate_proj": (C, I), "up_proj": (C, I), "down_proj": (I, C)}
    cast_bytes = 0.0
    out_bytes = 1.0 if lowp == "fp8" else 0.5
    for op in forward_ops(cfg, B, T):
        base = 3.0 * op.time(hw)
        t_bf16 += base
        kind = op.name.split(".", 1)[-1]
        if kind in LINEAR_OPS:
            K, N = shapes[kind]
            lp_bytes = out_bytes * (M * K + K * N) + 2.0 * M * N          # low-precision operands, BF16 output
            t = 3.0 * max(op.flops / lp.peak_flops, lp_bytes / hw.mem_bw)
            t_lin_bf16 += base
            t_lin_lp += t
            t_lp += t
            cast_bytes += (2 + out_bytes) * (2 * M * K + 2 * M * N + 2 * N * K)
        else:
            t_lp += base
    t_cast = 0.0 if fused_casts else cast_bytes / hw.mem_bw
    t_lp += t_cast
    return {"hw": hw.name, "t_bf16": t_bf16, "t_lowp": t_lp, "speedup": t_bf16 / t_lp,
            "linear_share_bf16": t_lin_bf16 / t_bf16, "t_linear_bf16": t_lin_bf16, "t_linear_lowp": t_lin_lp,
            "t_cast": t_cast, "amdahl_limit": 1.0 / (1.0 - t_lin_bf16 / t_bf16 * (1 - 1 / LOWP_FACTOR[lowp]))}
