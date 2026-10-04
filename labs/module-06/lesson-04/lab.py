"""Lab 06.4 — conditional memory (Engram, extension). Fill in the TODOs; run `pytest labs/module-06/lesson-04`."""

from __future__ import annotations

import torch

MOD = 2_147_483_647        # 2^31 - 1: keeps every product inside int64


def ngram_hash(ids: torch.Tensor, n: int, mult: int, mod: int, pad: int) -> torch.Tensor:
    """Hash the suffix n-gram ending at every position of ``ids`` (B, S), returning bucket ids (B, S) in [0, mod).

    Pad the left with n-1 copies of ``pad`` (positions near the start have fewer than n-1 earlier tokens).
    For each position, over the n-gram's tokens x (oldest first): h = ((h * mult) mod MOD) XOR (x + 1),
    starting from h = 0; the bucket is h mod ``mod``. It must equal frontierlab.blocks.engram.hash_ngrams.
    """
    raise NotImplementedError("TODO 1: the multiplicative-XOR n-gram hash")


def engram_gate(h: torch.Tensor, k: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Engram Eq. 4: alpha = sigmoid( RMSNorm(h) . RMSNorm(k) / sqrt(C) ), RMSNorm without a learned gain
    (x / sqrt(mean(x^2) + eps)). h, k (..., C) -> alpha (...)."""
    raise NotImplementedError("TODO 2: the context-aware gate")


def experts_at(rho: float, experts_at_one: int, top_k: int) -> int:
    """The Engram paper's allocation ratio rho (section 3.1): the fraction of the *inactive* parameter budget kept
    as routed experts. With ``experts_at_one`` experts at rho = 1 and ``top_k`` of them active per token, return
    the number of routed experts at ``rho`` (rounded). Check it against the paper's 46 / 43 / 55 experts."""
    raise NotImplementedError("TODO 3: experts at an allocation ratio")


def collision_free_share(distinct: int, table: int, heads: int = 1) -> float:
    """Under uniform hashing of ``distinct`` n-grams into ``heads`` independent tables of ``table`` buckets each:
    the expected share of n-grams that are alone in their bucket in at least one table.
    One table: an n-gram is alone with probability (1 - 1/M)^(N-1). With K independent tables it fails only if it
    shares a bucket in all K."""
    raise NotImplementedError("TODO 4: why Engram uses several hash heads")
