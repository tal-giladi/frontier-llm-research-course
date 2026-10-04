"""A model of limited-precision accumulation inside an FP8 GEMM (lesson 08.2).

DeepSeek-V3 section 3.3.2 ("Increasing Accumulation Precision"): the "accumulation precision of FP8 GEMM on NVIDIA
H800 GPUs is limited to retaining around 14 bits", which for K = 4096 random matrices gives "a maximum relative
error of nearly 2%". Their fix: accumulate N_C = 128 elements (4 WGMMAs) on the Tensor Cores, then promote the
partial sum to an FP32 register on the CUDA cores and add it there.

The hardware's exact accumulator behaviour is not public. This module is a **model** of it, labelled as such: the
running sum is rounded to ``mant_bits`` explicit mantissa bits (unlimited exponent) after every addition, either
to nearest or by truncation (the alignment shift of a fixed-width adder drops low bits; INFERENCE). It reproduces
the qualitative behaviour — error growing with K, and promotion every N_C elements bounding it — not the H800's
exact numbers. The products themselves are exact here: an E4M3 × E4M3 product has at most 8 significant bits.
"""

from __future__ import annotations

import torch


def round_mantissa(x: torch.Tensor, mant_bits: int, mode: str = "rne") -> torch.Tensor:
    """Round float64 ``x`` to ``mant_bits`` bits after the leading one (no exponent limit)."""
    _, e = torch.frexp(x)
    shift = (e - 1 - mant_bits).to(torch.int32)
    q = torch.ldexp(x, -shift)
    q = torch.round(q) if mode == "rne" else torch.trunc(q)
    return torch.ldexp(q, shift)


def limited_dot(a: torch.Tensor, b: torch.Tensor, mant_bits: int = 13, promote_every: int | None = None,
                mode: str = "trunc") -> torch.Tensor:
    """Dot products along the last dim of ``a`` (..., K) and ``b`` (..., K) with a limited accumulator.

    Products are summed in order into an accumulator rounded to ``mant_bits`` after each add. With
    ``promote_every = N_C`` the accumulator is added to an exact (float64) total every N_C elements and reset — the
    DeepSeek-V3 promotion. 13 explicit bits = 14 significant bits ("around 14 bits"); the mapping from the paper's
    wording to a bit count is an assumption.
    """
    p = (a.double() * b.double())
    K = p.shape[-1]
    total = torch.zeros(p.shape[:-1], dtype=torch.float64, device=p.device)
    acc = torch.zeros_like(total)
    for k in range(K):
        acc = round_mantissa(acc + p[..., k], mant_bits, mode)
        if promote_every and (k + 1) % promote_every == 0:
            total = total + acc
            acc = torch.zeros_like(acc)
    return total + acc


def accumulation_error(K: int, n: int = 256, mant_bits: int = 13, promote_every: int | None = None,
                       mode: str = "trunc", seed: int = 0, positive: bool = True) -> dict:
    """Relative error of ``n`` length-K dot products against float64, for random E4M3-quantised inputs.

    ``positive=True`` draws non-negative inputs (the sum grows like K, the worst case for a fixed-width
    accumulator; random-sign inputs grow like √K and lose less).
    """
    from frontierlab.precision.quant import qdq
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(n, K, generator=g, dtype=torch.float64)
    b = torch.rand(n, K, generator=g, dtype=torch.float64)
    if not positive:
        a, b = a * 2 - 1, b * 2 - 1
    a, b = qdq(a, "fp8-tensor-e4m3"), qdq(b, "fp8-tensor-e4m3")
    exact = (a * b).sum(-1)
    approx = limited_dot(a, b, mant_bits, promote_every, mode)
    rel = ((approx - exact).abs() / exact.abs().clamp_min(1e-300))
    return {"K": K, "mean_rel_err": rel.mean().item(), "max_rel_err": rel.max().item()}
