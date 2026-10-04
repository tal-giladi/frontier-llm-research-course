"""Lab 08.1 — number formats and scaling. Fill in the TODOs; run `pytest labs/module-08/lesson-01` to check.

Write these with plain torch operations. Do not call frontierlab.precision (the tests compare against it).
"""

from __future__ import annotations

import torch


def round_fp(x: torch.Tensor, exp_bits: int, man_bits: int, bias: int, max_normal: float) -> torch.Tensor:
    """Round every element of ``x`` to the nearest value of a binary float format (ties to even) and clamp the
    result to [-max_normal, max_normal] (saturation, as the training recipes do).

    E4M3 is (4, 3, 7, 448.0); E5M2 (5, 2, 15, 57344.0); E2M1 (2, 1, 1, 6.0).
    Recipe from the lesson: the binade of x is k = floor(log2|x|) (use torch.frexp: x = m * 2^e with
    0.5 <= |m| < 1, so k = e - 1), floored at the smallest normal exponent 1 - bias; the spacing there is
    2^(k - man_bits); divide by the spacing, round with torch.round (half to even), multiply back. Use
    torch.ldexp for the exact power-of-two scaling. Return float32 (float64 for a float64 input).
    """
    raise NotImplementedError("TODO 1: round to a float format by its binade and spacing")


def block_qdq(x: torch.Tensor, exp_bits: int, man_bits: int, bias: int, max_normal: float, block: int) -> torch.Tensor:
    """Quantise-dequantise a 2-D tensor (R, C) with one fp32 scale per 1 x ``block`` group of the last dim.

    Per group: s = amax / max_normal (s = 1 for an all-zero group); result = round_fp(x / s) * s.
    C is a multiple of ``block``. block = C gives per-row scaling.
    """
    raise NotImplementedError("TODO 2: per-block amax scaling")


def mx_shared_scale(amax: torch.Tensor, emax_elem: int) -> torch.Tensor:
    """OCP MX v1.0 section 6.3 scale: the largest power of two <= amax, divided by the largest power of two of
    the element format (2^emax_elem; emax_elem = 8 for E4M3, 2 for E2M1). ``amax`` > 0. Returns powers of two."""
    raise NotImplementedError("TODO 3: the E8M0 shared scale")


def nvfp4_qdq(x: torch.Tensor) -> torch.Tensor:
    """NVFP4 quantise-dequantise of a 2-D tensor (R, C), C a multiple of 16, blocks of 1 x 16 along the last dim.

    Two levels (arXiv 2509.25149 appendix B):
      tensor decode scale  t = amax(x) / (6 * 448)                        fp32
      block scale          b = round_fp(amax_block / 6 / t) in E4M3        (4, 3, 7, 448.0)
      element              q = round_fp(x / (b * t)) in E2M1              (2, 1, 1, 6.0)
      result               q * b * t      (a block whose b * t is 0 becomes all zeros)
    """
    raise NotImplementedError("TODO 4: two-level NVFP4 scaling")
