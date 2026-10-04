"""Lab 08.2 — FP8 training. Fill in the TODOs; run `pytest labs/module-08/lesson-02` to check.

You may use ``frontierlab.precision.round_to_format`` (lesson 08.1) and
``frontierlab.precision.accum.round_mantissa``; do not call ``frontierlab.precision.qdq`` (the tests compare with it).
"""

from __future__ import annotations

import torch

from frontierlab.precision import E4M3, round_to_format  # noqa: F401
from frontierlab.precision.accum import round_mantissa  # noqa: F401


def tile_qdq(x: torch.Tensor, tile: int = 128) -> torch.Tensor:
    """DeepSeek-V3 activation quantisation: E4M3 with one FP32 scale per 1 x ``tile`` group of the last dim
    (per token per 128 channels). s = amax / 448 (1 for an all-zero tile). C is a multiple of ``tile``."""
    raise NotImplementedError("TODO 1a: 1 x 128 tile quantisation")


def block_qdq(w: torch.Tensor, block: int = 128) -> torch.Tensor:
    """DeepSeek-V3 weight quantisation: E4M3 with one FP32 scale per ``block`` x ``block`` square block.
    Both dims are multiples of ``block``."""
    raise NotImplementedError("TODO 1b: 128 x 128 block quantisation")


def promoted_sum(products: torch.Tensor, mant_bits: int, n_c: int | None) -> torch.Tensor:
    """Sum ``products`` (..., K) in order with a limited accumulator: after every addition round the running sum
    with ``round_mantissa(acc, mant_bits, "trunc")``. If ``n_c`` is set, every ``n_c`` elements add the accumulator
    to an exact float64 total and reset it to zero (DeepSeek-V3's promotion to FP32 on CUDA cores, N_C = 128).
    Return total + the remaining accumulator, float64, shape (...)."""
    raise NotImplementedError("TODO 2: limited accumulation with promotion")


def fp8_backward(x: torch.Tensor, w: torch.Tensor, gy: torch.Tensor, tile: int = 128):
    """The two backward GEMMs of y = x Wᵀ under the DeepSeek-V3 recipe, x (M, K), w (N, K), gy (M, N).

    Each operand is quantised along the dimension its GEMM contracts over:
      Dgrad  dx = q(dy) q(W)          dy in 1 x 128 tiles along N; W in 128 x 128 blocks
      Wgrad  dW = q(dyᵀ) q(xᵀ)ᵀ       dyᵀ (N, M) and xᵀ (K, M) in 1 x 128 tiles along the token dim M
    Return (dx, dW).
    """
    raise NotImplementedError("TODO 3: the backward GEMMs on correctly oriented FP8 operands")


def decide(gap_ci: tuple[float, float], margin: float, speedup_ci: tuple[float, float] | None,
           per_seed: list[float] | None = None) -> str:
    """The contract's rule. ``gap_ci``: 95% interval of (FP8 loss - BF16 loss), paired; ``margin``: largest acceptable
    gap; ``speedup_ci``: interval of the BF16/FP8 step-time ratio from real kernels, or None if not measured.

      "reject"        if the whole gap interval is above the margin (lo > margin)
      "inconclusive"  otherwise, if ``per_seed`` (the gap of each seed pair) is given and any |gap| > margin:
                      the window interval does not include seed-to-seed variation, so it cannot be trusted alone
      "adopt"         if hi <= margin and the speed-up interval is entirely above 1
      "reject"        if hi <= margin but the speed-up interval is not entirely above 1 (no reason to switch)
      "numerics ok, speed not measured"   if hi <= margin and speedup_ci is None (emulation only)
      "inconclusive"  otherwise
    """
    raise NotImplementedError("TODO 4: the pre-stated decision rule")
