"""Reference solution for lab 08.2."""

from __future__ import annotations

import torch

from frontierlab.precision import E4M3, round_to_format
from frontierlab.precision.accum import round_mantissa


def tile_qdq(x: torch.Tensor, tile: int = 128) -> torch.Tensor:
    R, C = x.shape
    xb = x.reshape(R, C // tile, tile)
    s = xb.abs().amax(-1, keepdim=True) / 448.0
    s = torch.where(s > 0, s, torch.ones_like(s))
    return (round_to_format(xb / s, E4M3) * s).reshape(R, C).to(x.dtype)


def block_qdq(w: torch.Tensor, block: int = 128) -> torch.Tensor:
    R, C = w.shape
    wb = w.reshape(R // block, block, C // block, block)
    s = wb.abs().amax(dim=(1, 3), keepdim=True) / 448.0
    s = torch.where(s > 0, s, torch.ones_like(s))
    return (round_to_format(wb / s, E4M3) * s).reshape(R, C).to(w.dtype)


def promoted_sum(products: torch.Tensor, mant_bits: int, n_c: int | None) -> torch.Tensor:
    total = torch.zeros(products.shape[:-1], dtype=torch.float64)
    acc = torch.zeros_like(total)
    for k in range(products.shape[-1]):
        acc = round_mantissa(acc + products[..., k], mant_bits, "trunc")
        if n_c and (k + 1) % n_c == 0:
            total, acc = total + acc, torch.zeros_like(acc)
    return total + acc


def fp8_backward(x: torch.Tensor, w: torch.Tensor, gy: torch.Tensor, tile: int = 128):
    gq = tile_qdq(gy, tile)                          # dy (M, N), tiles along N: Dgrad contracts over N
    wq = block_qdq(w, tile)                          # 128 x 128 blocks: W and W^T quantise identically
    dx = gq @ wq
    gt = tile_qdq(gy.t().contiguous(), tile)         # dy^T (N, M): Wgrad contracts over tokens M
    xt = tile_qdq(x.t().contiguous(), tile)          # x^T (K, M): the 128 x 1 tiles of DeepSeek-V3 B.2
    dw = gt @ xt.t()
    return dx, dw


def decide(gap_ci: tuple[float, float], margin: float, speedup_ci: tuple[float, float] | None,
           per_seed: list[float] | None = None) -> str:
    lo, hi = gap_ci
    if lo > margin:
        return "reject"
    if per_seed is not None and any(abs(g) > margin for g in per_seed):
        return "inconclusive"
    if hi <= margin:
        if speedup_ci is None:
            return "numerics ok, speed not measured"
        return "adopt" if speedup_ci[0] > 1.0 else "reject"
    return "inconclusive"
