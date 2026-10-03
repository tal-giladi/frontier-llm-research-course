"""Reference solution for lab 02.1."""

from __future__ import annotations

from frontierlab.flops import flops_per_token
from frontierlab.model.config import ModelConfig


def gemm_cost(M: int, N: int, K: int, bytes_per: int = 2) -> tuple[float, float]:
    return 2.0 * M * N * K, float(bytes_per) * (M * K + K * N + M * N)


def rmsnorm_cost(rows: int, C: int, bytes_per: int = 2) -> tuple[float, float]:
    return 4.0 * rows * C, 2.0 * bytes_per * rows * C


def softmax_cost(rows: int, cols: int, bytes_per: int = 2) -> tuple[float, float]:
    return 5.0 * rows * cols, 2.0 * bytes_per * rows * cols


def decode_attention_cost(B: int, H: int, KV: int, S: int, d: int, bytes_per: int = 2) -> tuple[float, float]:
    return 4.0 * B * H * S * d, 2.0 * bytes_per * B * KV * S * d


def roofline_time(flops: float, nbytes: float, peak_flops: float, mem_bw: float) -> float:
    return max(flops / peak_flops, nbytes / mem_bw)


def predict_step_time(cfg: ModelConfig, B: int, T: int, peak_flops: float, mfu: float, grad_accum: int = 1) -> float:
    return flops_per_token(cfg, T) * B * T * grad_accum / (peak_flops * mfu)
