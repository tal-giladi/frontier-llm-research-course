"""Lab 02.1 — from FLOPs to time. Fill in the TODOs; run `pytest labs/module-02/lesson-01` to check.

Every cost function returns ``(flops, bytes)``: the floating-point operations of the op and the
bytes it must move between memory and the chip if each input is read once and each output written
once. ``bytes_per`` is the element size (2 for bf16, 4 for fp32).
"""

from __future__ import annotations

from frontierlab.flops import flops_per_token  # noqa: F401  (you will want it in TODO 6)
from frontierlab.model.config import ModelConfig


def gemm_cost(M: int, N: int, K: int, bytes_per: int = 2) -> tuple[float, float]:
    """(M, K) @ (K, N) -> (M, N). One multiply-add = 2 FLOPs. Reads A and B, writes C."""
    raise NotImplementedError("TODO 1: FLOPs and bytes of a GEMM")


def rmsnorm_cost(rows: int, C: int, bytes_per: int = 2) -> tuple[float, float]:
    """RMSNorm over rows of width C. Count 4 FLOPs per element; read x, write y; ignore the gain."""
    raise NotImplementedError("TODO 2: FLOPs and bytes of RMSNorm")


def softmax_cost(rows: int, cols: int, bytes_per: int = 2) -> tuple[float, float]:
    """Row softmax. Count 5 FLOPs per element (max, subtract, exp, sum, divide); read x, write y."""
    raise NotImplementedError("TODO 3: FLOPs and bytes of softmax")


def decode_attention_cost(B: int, H: int, KV: int, S: int, d: int, bytes_per: int = 2) -> tuple[float, float]:
    """One new query per sequence (B sequences, H query heads, KV key/value heads, head dim d)
    attending to S cached tokens. FLOPs of q·Kᵀ and p·V; bytes of reading the K and V caches only."""
    raise NotImplementedError("TODO 4: FLOPs and bytes of attention at decode")


def roofline_time(flops: float, nbytes: float, peak_flops: float, mem_bw: float) -> float:
    """The shortest possible run time on a machine with this peak (FLOP/s) and bandwidth (bytes/s)."""
    raise NotImplementedError("TODO 5: the roofline lower bound")


def predict_step_time(cfg: ModelConfig, B: int, T: int, peak_flops: float, mfu: float, grad_accum: int = 1) -> float:
    """Seconds per optimizer step: training FLOPs of B·T·grad_accum tokens ÷ (peak × MFU)."""
    raise NotImplementedError("TODO 6: predicted step time")
