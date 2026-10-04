"""Lab 06.2 — hyper-connections and mHC. Fill in the TODOs; run `pytest labs/module-06/lesson-02` to check."""

from __future__ import annotations

import torch


def sinkhorn(logits: torch.Tensor, t_max: int = 20) -> torch.Tensor:
    """Sinkhorn-Knopp as in mHC Eq. 9, for logits of shape (..., n, n).

    M0 = exp(logits) — subtract each matrix's maximum first (why is that allowed?) — then t_max times:
    normalise every column to sum 1, then every row to sum 1. Return M(t_max).
    """
    raise NotImplementedError("TODO 1: Sinkhorn-Knopp")


def mhc_mix(X: torch.Tensor, F_sub, pre: torch.Tensor, post: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
    """One wrapped sub-layer, per token (mHC Eq. 3): x_{l+1} = H_res x_l + H_post^T F(H_pre x_l).

    X (B, T, n, C) is the n-stream residual; pre and post (B, T, n); res (B, T, n, n) with
    output stream i = sum_j res[i, j] · stream j. F_sub maps (B, T, C) -> (B, T, C).
    """
    raise NotImplementedError("TODO 2: read with H_pre, mix with H_res, write with H_post")


def composite_gain(res_list: list[torch.Tensor]) -> tuple[float, float]:
    """The Amax gain of mHC section 3.1 for the composite map H_res^(L) ... H_res^(2) H_res^(1).

    Each entry of res_list is (B, T, n, n), bottom layer first. Per token, multiply them in order (the newest on
    the left), then take the maximum absolute row sum (forward gain) and the maximum absolute column sum
    (backward gain); return both averaged over tokens, in float64.
    """
    raise NotImplementedError("TODO 3: composite forward and backward gain")


def residual_io(n: int, C: int) -> dict:
    """Elements read and written per token and sub-layer by the residual itself (mHC Table 2, forward, without F).

    n = 1: the plain residual, {"read": 2C, "write": C}. n > 1: sum the rows of Table 2 — computing the three
    mappings (read nC, write n² + 2n), applying H_pre (read nC + n, write C), H_post (read C + n, write nC),
    H_res (read nC + n², write nC) and the residual merge (read 2nC, write nC).
    """
    raise NotImplementedError("TODO 4: the memory traffic of the n-stream residual")
