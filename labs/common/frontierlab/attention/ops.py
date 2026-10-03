"""Attention arithmetic shared by the Module 3 kinds: banded causal masks, learned sinks, probabilities.

One function, :func:`attend`, computes causal attention for every Module 3 kind::

    y = attend(q, k, v, q_pos, k_pos, window=None, sink=None, scale=None)

    q        (B, H,  T, d_qk)      queries for T new tokens at absolute positions q_pos (T,)
    k        (B, KV, S, d_qk)      keys at absolute positions k_pos (S,); H must be a multiple of KV
    v        (B, KV, S, d_v)       values (d_v may differ from d_qk, as in MLA)
    window   int | None            query at p sees keys at positions p - window + 1 .. p (window keys,
                                   itself included); None = full causal attention
    sink     (H,) | None           one learned logit per head that joins the softmax denominator but has
                                   no value (gpt-oss model card, section 2.2)
    y        (B, H, T, d_v)

Window convention: ``window = w`` means a query sees ``w`` keys including itself, the convention of
Hugging Face Transformers' sliding-window masks (``kv_idx > q_idx - sliding_window``) and of
``frontierlab.calc`` (``min(S, w)`` keys at decode). Some code bases count ``w`` keys *before* the
query (``w + 1`` in total); check before you compare a window size across implementations.

Without a sink the work goes to ``F.scaled_dot_product_attention`` with a boolean mask. With a sink,
or when probabilities are requested, it runs an explicit softmax so that every step is visible.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def band_mask(q_pos: torch.Tensor, k_pos: torch.Tensor, window: int | None = None) -> torch.Tensor:
    """Boolean (T, S) mask, True where the query at ``q_pos[i]`` may attend the key at ``k_pos[j]``."""
    m = k_pos[None, :] <= q_pos[:, None]
    if window is not None:
        m = m & (k_pos[None, :] > q_pos[:, None] - window)
    return m


def _expand_kv(x: torch.Tensor, H: int) -> torch.Tensor:
    """(B, KV, S, d) -> (B, H, S, d) by repeating each KV head H // KV times (GQA)."""
    rep = H // x.shape[1]
    return x if rep == 1 else x.repeat_interleave(rep, dim=1)


def attention_logits(q, k, q_pos, k_pos, window=None, scale=None):
    """Masked, scaled logits (B, H, T, S); masked entries are -inf."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    logits = (q @ _expand_kv(k, q.shape[1]).transpose(-1, -2)) * scale
    return logits.masked_fill(~band_mask(q_pos, k_pos, window), float("-inf"))


def sink_softmax(logits: torch.Tensor, sink: torch.Tensor | None) -> torch.Tensor:
    """Softmax over the last axis with one extra per-head logit that takes probability but returns no value.

    logits (B, H, T, S), sink (H,). Returns (B, H, T, S); each row sums to 1 - p_sink <= 1.
    Computed in at least float32 (then cast back) so a bf16 row does not lose its small entries.
    """
    work = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
    if sink is None:
        return torch.softmax(logits.to(work), dim=-1).to(logits.dtype)
    B, H, T, _ = logits.shape
    s = sink.to(work).view(1, H, 1, 1).expand(B, H, T, 1)
    p = torch.softmax(torch.cat((logits.to(work), s), dim=-1), dim=-1)
    return p[..., :-1].to(logits.dtype)


def attend(q, k, v, q_pos, k_pos, *, window: int | None = None, sink: torch.Tensor | None = None,
           scale: float | None = None, return_probs: bool = False):
    """Causal (optionally banded, optionally sink-augmented) attention; see the module docstring."""
    H, KV = q.shape[1], k.shape[1]
    if H % KV:
        raise ValueError("query heads must be a multiple of key/value heads")
    if sink is None and not return_probs:
        square = q.shape[2] == k.shape[2] and torch.equal(q_pos, k_pos)
        if square and window is None:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=H != KV)
        mask = band_mask(q_pos, k_pos, window)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=H != KV)
    p = sink_softmax(attention_logits(q, k, q_pos, k_pos, window, scale), sink)
    y = p @ _expand_kv(v, H)
    return (y, p) if return_probs else y
