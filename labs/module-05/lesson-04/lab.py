"""Lab 05.4 (extension) — compressed KV entries in the style of DeepSeek-V4's CSA and HCA, and their cost.

Fill in the TODOs; run `pytest labs/module-05/lesson-04`. Reference: frontierlab.attention.compressed.
"""

from __future__ import annotations

import torch


def compress_hca(c: torch.Tensor, z: torch.Tensor, m: int) -> torch.Tensor:
    """c, z (B, S, d) -> (B, S // m, d). Entry i = sum over the m tokens j of block i of softmax_j(z_j) * c_j,
    the softmax taken over the block, separately for every channel. Drop an incomplete last block."""
    raise NotImplementedError("TODO 1: non-overlapping compression (HCA style)")


def compress_csa(ca, za, cb, zb, m: int) -> torch.Tensor:
    """Overlapping compression: entry i = sum_{j in block i} S^a_j c^a_j + sum_{j in block i-1} S^b_j c^b_j,
    S = softmax over the 2m logits (za over block i, zb over block i-1), per channel. Entry 0 has no previous
    block: only path a, with its softmax over m logits."""
    raise NotImplementedError("TODO 2: overlapping compression (CSA style)")


def usable_entries(t: int, m: int) -> int:
    """How many compressed entries may a query at position t (0-based) use without seeing the future?"""
    raise NotImplementedError("TODO 3: causality of compressed entries")


def layout_kv_bytes(S: int, layers: list[str], *, m: int, m2: int, window: int, entry_bytes: float) -> float:
    """KV bytes of one sequence after S tokens for a list of layer kinds "csa" / "hca" / "dense".

    csa: S // m compressed entries + min(S, window) uncompressed recent tokens; hca: S // m2 entries;
    dense: S entries. Every entry (compressed or not) costs ``entry_bytes``.
    """
    raise NotImplementedError("TODO 4: KV bytes of a compressed layout")
