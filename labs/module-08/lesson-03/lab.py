"""Lab 08.3 — FP4 training and quantisation-aware training. Fill in the TODOs; run `pytest labs/module-08/lesson-03`.

Plain torch only; the tests compare against frontierlab.precision.
"""

from __future__ import annotations

import torch

E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def sr_round_e2m1(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Stochastic rounding of already-scaled values ``x`` to the E2M1 grid ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}.

    |x| is first clamped to 6. With neighbours lo <= |x| <= hi on the grid, round up to hi exactly when
    u < (|x| - lo) / (hi - lo), where ``u`` (same shape as x) holds uniform [0, 1) numbers; keep the sign.
    Values on the grid stay where they are. (torch.searchsorted finds the neighbours.)
    """
    raise NotImplementedError("TODO 1: stochastic rounding to the E2M1 grid")


def hadamard16(dtype=torch.float64) -> torch.Tensor:
    """The 16 x 16 Sylvester Hadamard matrix divided by 4 (orthonormal rows)."""
    raise NotImplementedError("TODO 2a: build H16")


def rht_blocks(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Random Hadamard transform of the last dim of ``x`` in blocks of 16: each block v becomes v @ (H16 · diag(signs)).
    ``signs`` is a (16,) tensor of ±1. The last dim is a multiple of 16."""
    raise NotImplementedError("TODO 2b: blockwise random Hadamard transform")


def int4_group_qdq(w: torch.Tensor, group: int = 32) -> torch.Tensor:
    """Weight-only symmetric INT4 with one scale per ``group`` consecutive input channels (the grouping of
    Kimi-K2-Thinking's config.json): s = amax / 7 (1 for an all-zero group), q = clamp(round(w / s), -7, 7)."""
    raise NotImplementedError("TODO 3: grouped INT4")


def qat_backward(x: torch.Tensor, w: torch.Tensor, gy: torch.Tensor, group: int = 32):
    """Backward of y = x · q(W)ᵀ in quantisation-aware training with the straight-through estimator.
    x (M, K), w (N, K), gy (M, N). Return (dx, dW): dx through the quantised weight the forward used; dW as if
    q were the identity."""
    raise NotImplementedError("TODO 4: the STE backward")
