"""Reference solution for lab 06.2 (hyper-connections and mHC)."""

from __future__ import annotations

import torch


def sinkhorn(logits: torch.Tensor, t_max: int = 20) -> torch.Tensor:
    """mHC Eq. 9: M0 = exp(logits); t_max times normalise columns, then rows. logits (..., n, n)."""
    m = torch.exp(logits - logits.amax(dim=(-2, -1), keepdim=True))
    for _ in range(t_max):
        m = m / m.sum(-2, keepdim=True)
        m = m / m.sum(-1, keepdim=True)
    return m


def mhc_mix(X: torch.Tensor, F_sub, pre: torch.Tensor, post: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
    """x_{l+1} = H_res x_l + H_post^T F(H_pre x_l), per token. X (B, T, n, C); pre, post (B, T, n); res (B, T, n, n)."""
    h_in = torch.einsum("btj,btjc->btc", pre, X)
    y = F_sub(h_in)
    return torch.einsum("btij,btjc->btic", res, X) + post.unsqueeze(-1) * y.unsqueeze(-2)


def composite_gain(res_list: list[torch.Tensor]) -> tuple[float, float]:
    """Forward / backward Amax gain of the composite H_res^(L) ... H_res^(1), per token, averaged over tokens:
    max absolute row sum (forward) and max absolute column sum (backward). Each entry (B, T, n, n)."""
    comp = None
    for m in res_list:
        m = m.double()
        comp = m if comp is None else m @ comp
    fwd = comp.abs().sum(-1).amax(-1).mean().item()
    bwd = comp.abs().sum(-2).amax(-1).mean().item()
    return fwd, bwd


def residual_io(n: int, C: int) -> dict:
    """mHC Table 2, per token and sub-layer, forward, excluding F: elements read and written by the residual.
    n = 1 means the plain residual (2C read, C written)."""
    if n == 1:
        return {"read": 2 * C, "write": C}
    return {"read": (5 * n + 1) * C + n * n + 2 * n, "write": (3 * n + 1) * C + n * n + 2 * n}
