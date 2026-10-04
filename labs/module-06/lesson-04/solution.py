"""Reference solution for lab 06.4 (Engram, extension)."""

from __future__ import annotations

import torch

MOD = 2_147_483_647


def ngram_hash(ids: torch.Tensor, n: int, mult: int, mod: int, pad: int) -> torch.Tensor:
    """Multiplicative-XOR hash of the suffix n-gram at every position (B, S) -> (B, S) in [0, mod)."""
    B, S = ids.shape
    padded = torch.cat([torch.full((B, n - 1), pad, dtype=ids.dtype), ids], 1)
    h = torch.zeros(B, S, dtype=torch.int64)
    for j in range(n):
        h = torch.bitwise_xor((h * mult) % MOD, padded[:, j:j + S].to(torch.int64) + 1)
    return h % mod


def engram_gate(h: torch.Tensor, k: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """alpha = sigmoid( RMSNorm(h) . RMSNorm(k) / sqrt(C) ), RMSNorm without a gain; h, k (..., C) -> (...)."""
    C = h.shape[-1]

    def rms(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    return torch.sigmoid((rms(h) * rms(k)).sum(-1) / C ** 0.5)


def experts_at(rho: float, experts_at_one: int, top_k: int) -> int:
    """Routed experts when a fraction rho of the inactive (unselected-expert) budget stays with MoE.
    At rho = 1 there are ``experts_at_one`` experts, of which ``top_k`` are active per token."""
    return top_k + round(rho * (experts_at_one - top_k))


def collision_free_share(distinct: int, table: int, heads: int = 1) -> float:
    """Expected share of distinct n-grams that share their bucket with no other n-gram in at least one of
    ``heads`` independent hash tables of size ``table`` (uniform hashing): 1 − (1 − (1 − 1/M)^(N−1))^K."""
    p_alone = (1 - 1 / table) ** (distinct - 1)
    return 1 - (1 - p_alone) ** heads
