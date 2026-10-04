"""Roofline arithmetic: how long an operation *must* take on a given GPU (Module 2, lesson 02.1).

An operation does F floating-point operations and moves Q bytes between HBM and the chip. A GPU
with peak P FLOP/s and memory bandwidth W bytes/s cannot finish it faster than

    t_min = max(F / P, Q / W)

Arithmetic intensity I = F / Q (FLOP per byte). The ridge point P / W is the intensity at which the
two limits are equal: below it an op is memory-bound, above it compute-bound.

All peaks are DENSE (no 2:4 sparsity). NVIDIA's datasheets list BF16/FP16 tensor-core numbers
"with sparsity"; the dense number is half. Values checked on the vendor pages on 2026-10-03:

    H100 SXM 80GB   BF16 1,979 TFLOP/s with sparsity -> 989 dense;  HBM3 3.35 TB/s
    A100 SXM 80GB   BF16 312 dense (624 with sparsity);              HBM2e 2,039 GB/s
    L4              BF16 242 with sparsity -> 121 dense;             GDDR6 300 GB/s
    T4              FP16 65 TFLOP/s mixed precision (no BF16 tensor cores); GDDR6 320 GB/s

Real kernels reach a fraction of either peak; this module gives lower bounds and a step-time model,
never a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

from frontierlab.flops import flops_per_token
from frontierlab.model.config import ModelConfig


@dataclass(frozen=True)
class Hardware:
    name: str
    peak_flops: float        # dense tensor-core FLOP/s at the training dtype
    mem_bw: float            # HBM / GDDR bytes per second
    dtype: str               # the dtype the peak refers to
    source: str

    @property
    def ridge(self) -> float:
        """FLOP per byte at which compute and memory limits meet."""
        return self.peak_flops / self.mem_bw


HARDWARE = {
    "H100-SXM": Hardware("H100 SXM5 80GB", 989e12, 3.35e12, "bf16",
                         "https://www.nvidia.com/en-us/data-center/h100/"),
    "A100-SXM-80GB": Hardware("A100 SXM4 80GB", 312e12, 2.039e12, "bf16",
                              "https://www.nvidia.com/en-us/data-center/a100/"),
    "A100-SXM-40GB": Hardware("A100 SXM4 40GB (the usual Colab A100)", 312e12, 1.555e12, "bf16",
                              "https://www.nvidia.com/en-us/data-center/a100/"),
    "B200": Hardware("B200 (per GPU: DGX B200 figures / 8)", 2.25e15, 8.0e12, "bf16",
                     "https://www.nvidia.com/en-us/data-center/dgx-b200/"),
    "L4": Hardware("L4 24GB", 121e12, 0.300e12, "bf16", "https://www.nvidia.com/en-us/data-center/l4/"),
    "T4": Hardware("T4 16GB", 65e12, 0.320e12, "fp16", "https://www.nvidia.com/en-us/data-center/tesla-t4/"),
}


@dataclass(frozen=True)
class Op:
    """One operation: its FLOPs and the bytes it must move (each input read once, output written once)."""
    name: str
    flops: float
    bytes: float

    @property
    def intensity(self) -> float:
        return self.flops / self.bytes if self.bytes else float("inf")

    def time(self, hw: Hardware) -> float:
        return roofline_time(self.flops, self.bytes, hw)

    def bound(self, hw: Hardware) -> str:
        return "compute" if self.intensity >= hw.ridge else "memory"


def roofline_time(flops: float, nbytes: float, hw: Hardware) -> float:
    """Lower bound on run time: max(F / peak, Q / bandwidth)."""
    return max(flops / hw.peak_flops, nbytes / hw.mem_bw)


def attainable_flops(intensity: float, hw: Hardware) -> float:
    """The roofline itself: min(peak, I × bandwidth)."""
    return min(hw.peak_flops, intensity * hw.mem_bw)


# ---- the four worked examples of lesson 02.1 -------------------------------------------------

def gemm(M: int, N: int, K: int, bytes_per: int = 2, name: str = "gemm") -> Op:
    """(M, K) @ (K, N): 2·M·N·K FLOPs; reads A and B, writes C."""
    return Op(name, 2.0 * M * N * K, float(bytes_per) * (M * K + K * N + M * N))


def rmsnorm(rows: int, C: int, bytes_per: int = 2, name: str = "rmsnorm") -> Op:
    """y = x / rms(x) · g per row. ~4 FLOPs per element (square, add, scale, gain); reads x, writes y.

    The gain vector (C elements) is negligible next to rows × C and is ignored.
    """
    return Op(name, 4.0 * rows * C, 2.0 * bytes_per * rows * C)


def softmax(rows: int, cols: int, bytes_per: int = 2, name: str = "softmax") -> Op:
    """Row softmax. ~5 FLOPs per element (max, subtract, exp, sum, divide); reads x, writes y."""
    return Op(name, 5.0 * rows * cols, 2.0 * bytes_per * rows * cols)


def decode_attention(B: int, H: int, KV: int, S: int, d: int, bytes_per: int = 2,
                     name: str = "decode-attention") -> Op:
    """One new query token per sequence attending to S cached tokens.

    FLOPs: q·Kᵀ and p·V are each 2·H·S·d per sequence -> 4·B·H·S·d.
    Bytes: the K and V caches, 2·B·KV·S·d elements (q and the output are tiny and ignored).
    Intensity = 4·H·S·d / (2·KV·S·d·bytes) = 2·(H/KV)/bytes FLOP per byte, independent of S and B.
    """
    return Op(name, 4.0 * B * H * S * d, 2.0 * bytes_per * B * KV * S * d)


# ---- a per-op model of one Baseline-0 training step ---------------------------------------------

def forward_ops(cfg: ModelConfig, B: int, T: int, bytes_per: int = 2, logits_bytes: int = 4) -> list[Op]:
    """The forward pass of one micro-batch as a list of ops (one layer's ops repeated L times).

    Assumes a fused (FlashAttention-style) attention kernel, so the T × T score matrix is never
    written to HBM: its bytes are q, k, v and the output. Elementwise ops (RoPE, SiLU·mul, residual
    adds) are counted as unfused reads and writes. The output head writes fp32 logits and the
    cross-entropy reads them back, as ``frontierlab.model.LM`` does.
    """
    C, L, H, KV, d, I, V = (cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads,
                            cfg.num_key_value_heads, cfg.head_dim, cfg.intermediate_size, cfg.vocab_size)
    M = B * T                                    # tokens in the micro-batch
    b = bytes_per
    layer = [
        rmsnorm(M, C, b, "input_layernorm"),
        gemm(M, H * d, C, b, "q_proj"),
        gemm(M, KV * d, C, b, "k_proj"),
        gemm(M, KV * d, C, b, "v_proj"),
        Op("qk_norm+rope", 10.0 * M * (H + KV) * d, 2.0 * b * M * (H + KV) * d),
        # causal attention: QKᵀ and PV each 2·T·T·d per head, halved by the mask
        Op("attention(flash)", 2.0 * B * H * T * T * d, float(b) * M * (2 * H * d + 2 * KV * d)),
        gemm(M, C, H * d, b, "o_proj"),
        Op("residual_add", 1.0 * M * C, 3.0 * b * M * C),
        rmsnorm(M, C, b, "post_attention_layernorm"),
        gemm(M, I, C, b, "gate_proj"),
        gemm(M, I, C, b, "up_proj"),
        Op("silu*mul", 5.0 * M * I, 3.0 * b * M * I),
        gemm(M, C, I, b, "down_proj"),
        Op("residual_add", 1.0 * M * C, 3.0 * b * M * C),
    ]
    ops = [Op(f"L{i}.{o.name}", o.flops, o.bytes) for i in range(L) for o in layer]
    ops += [rmsnorm(M, C, b, "final_norm"),
            Op("lm_head", 2.0 * M * V * C, float(b) * (M * C + V * C) + logits_bytes * M * V),
            Op("cross_entropy", 5.0 * M * V, float(logits_bytes) * M * V)]
    return ops


def step_time_model(cfg: ModelConfig, B: int, T: int, hw: Hardware, grad_accum: int = 1,
                    bytes_per: int = 2) -> dict:
    """Roofline lower bound for one optimizer step, op by op (forward + backward).

    Backward is modelled as 2× the forward FLOPs and 2× the forward bytes of every op (each op reads
    its saved inputs and the incoming gradient, writes input gradients and, for GEMMs, weight
    gradients). The optimizer update (AdamW reads/writes ~16 bytes of fp32 state per parameter) is
    added once per step. Result keys: ``t_compute_bound`` (GEMM + attention ops), ``t_memory_bound``
    (all other ops), ``t_optimizer``, ``t_total`` and ``ops`` (forward ops, for inspection).
    """
    ops = forward_ops(cfg, B, T, bytes_per)
    t_c = t_m = 0.0
    for o in ops:
        t = 3.0 * o.time(hw)
        if o.bound(hw) == "compute":
            t_c += t
        else:
            t_m += t
    from frontierlab.model.config import param_counts
    n_params = param_counts(cfg)["total"]
    t_opt = 16.0 * n_params / hw.mem_bw
    t_c, t_m = t_c * grad_accum, t_m * grad_accum
    return {"t_compute_bound": t_c, "t_memory_bound": t_m, "t_optimizer": t_opt,
            "t_total": t_c + t_m + t_opt, "ops": ops}


def predicted_step_time(cfg: ModelConfig, B: int, T: int, hw: Hardware, mfu: float,
                        grad_accum: int = 1) -> float:
    """The simple prediction: training FLOPs of the step ÷ (peak × assumed MFU)."""
    tokens = B * T * grad_accum
    return flops_per_token(cfg, T) * tokens / (hw.peak_flops * mfu)
