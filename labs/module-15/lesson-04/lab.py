"""Lab 15.4 — KV and weight quantisation for serving. Fill in the TODOs; run `pytest labs/module-15/lesson-04`.

``quant_lab.py`` checks your quantiser against the course's (``frontierlab.ttc.kvquant``) and then measures what
each KV and weight format does to the 15.2 target model's next-token distributions.
"""

from __future__ import annotations

import torch


def asym_qdq(x: torch.Tensor, bits: int, dim: int, group: int) -> torch.Tensor:
    """Quantise-dequantise in groups of ``group`` consecutive elements along ``dim``: per group
    s = (max - min) / (2^bits - 1), q = clamp(round((x - min) / s), 0, 2^bits - 1), return q * s + min.
    A group whose max equals its min is returned unchanged. The size of ``dim`` is a multiple of ``group``."""
    raise NotImplementedError("TODO 1: asymmetric group quantisation")


def kivi_qdq(k: torch.Tensor, v: torch.Tensor, bits: int, group: int, residual: int):
    """k, v (B, KV, S, d). Quantise only the first n_q = max(0, (S - residual) // group) * group tokens:
    keys PER CHANNEL (groups of ``group`` tokens along dim 2), values PER TOKEN (groups of ``group`` channels
    along dim 3, or the whole d if d is not a multiple of ``group``). The rest stays exact. Return (k', v')."""
    raise NotImplementedError("TODO 2: KIVI's key/value layout with a residual window")


def kv_cache_bytes(S: int, elements_per_token: int, bits: int, group: int, residual: int, full_bits: int = 16) -> float:
    """Bytes of one sequence's cache in one layer: quantised tokens cost bits + 32/group bits per element (a
    16-bit scale and a 16-bit zero point per group); the tokens that are not quantised cost full_bits."""
    raise NotImplementedError("TODO 3: bytes of a quantised cache")
