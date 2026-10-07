"""Lab 15.3 — Serving cost of architecture choices. Fill in the TODOs; run `pytest labs/module-15/lesson-03`.

``serve_lab.py`` prints your numbers next to the course's (``frontierlab.ttc.serving``, which reuses the exact
Module 3–5 cache accounting) for every design at 128K.
"""

from __future__ import annotations


def kv_bytes_per_token(layers: int, kind: str, *, kv_heads: int = 0, head_dim: int = 0, d_c: int = 0, d_r: int = 0,
                       bytes_per: float = 2) -> float:
    """Cache bytes per token for a model whose layers all use one global attention kind.
    kind "gqa": keys and values, kv_heads x head_dim each, per layer. kind "mla": one latent of d_c plus the
    shared RoPE key of d_r per layer (DeepSeek-V2 section 2.1)."""
    raise NotImplementedError("TODO 1: KV bytes per token, GQA and MLA")


def local_global_bytes(S: int, n_local: int, n_global: int, window: int, kv_heads: int, head_dim: int,
                       bytes_per: float = 2) -> float:
    """Cache bytes of ONE sequence after S tokens: global layers keep all S tokens, local (sliding-window)
    layers keep at most ``window``."""
    raise NotImplementedError("TODO 2: KV bytes of a local/global layout")


def decode_step_time(weight_bytes: float, kv_bytes_per_seq: float, batch: int, flops_per_token: float,
                     peak_flops: float, mem_bw: float) -> float:
    """Roofline lower bound of one decode step: every weight read once, every sequence's cache read once,
    flops_per_token FLOPs per sequence: max(compute time, memory time)."""
    raise NotImplementedError("TODO 3: decode step on the roofline")


def capacity(hbm_bytes: float, weight_bytes: float, kv_bytes_per_seq: float, reserve: float = 0.10) -> int:
    """Number of sequences whose caches fit next to the weights when ``reserve`` of HBM is kept free."""
    raise NotImplementedError("TODO 4: batch capacity")
