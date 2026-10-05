"""Pretraining packing with and without document-boundary masking (lesson 10.1).

Packing (parent course lesson 04.3) concatenates documents, separated by ``<|endoftext|>``, and cuts the
stream into fixed windows. Without a mask, a token attends to every earlier token of its window,
including tokens of unrelated earlier documents. **Document masking** (block-diagonal causal
attention; parent course lesson 14.2; Llama 3 report section 3.2) lets a token attend only to earlier
tokens of its own document:

    allowed[i, j] = (k_pos[j] <= q_pos[i]) and (seg[k_pos[j]] == seg[q_pos[i]])

where ``seg`` is the document index of every position of the window (:func:`segment_ids`).

The course attention interface carries one ``positions`` vector per batch and no per-sequence data,
so the segment ids travel through a context variable instead:

    with document_segments(seg):            # seg (B, T_total) int64, document index per position
        loss = model(x, labels=x).loss       # every "gqa-docmask" layer reads seg

:class:`DocMaskGQAttention` (registered as ``"gqa-docmask"``) is Baseline-0's attention plus that mask.
It has exactly GQA's parameters (a ``"gqa"`` checkpoint loads into it with ``strict=True``), and with
no segments set it computes exactly what ``"gqa"`` computes. The segments are indexed by *absolute*
position, so cached decoding works too: the new queries look up ``seg[:, positions]`` and the cached
keys ``seg[:, cache_positions]``.

Why the mask needs no position reset: RoPE makes ``q_i · k_j`` depend only on ``i - j``, so a document
that starts at window offset 300 produces the same attention scores among its own tokens as if it
started at 0. Packed + masked logits therefore equal running every document on its own (tested in
float64 to ~1e-12 in ``tests/test_datax.py``).

Cost. SDPA with a boolean mask still computes every (query, key) score and then discards the masked
ones; the FLOPs saved by block-diagonal attention appear only with kernels that skip whole masked
blocks (FlexAttention block masks, FlashAttention's variable-length ``cu_seqlens`` API).
:func:`useful_pair_fraction` measures how much of the causal score matrix a document mask keeps.
"""

from __future__ import annotations

import contextlib
import contextvars

import numpy as np
import torch
import torch.nn.functional as F

from frontierlab.attention.base import apply_rope, register
from frontierlab.attention.gqa import GQAttention

_SEGMENTS: contextvars.ContextVar[torch.Tensor | None] = contextvars.ContextVar("datax_segments", default=None)


@contextlib.contextmanager
def document_segments(seg: torch.Tensor | None):
    """Make ``seg`` (B, T_total) the document ids that ``"gqa-docmask"`` layers use inside the block."""
    token = _SEGMENTS.set(seg)
    try:
        yield
    finally:
        _SEGMENTS.reset(token)


def current_segments() -> torch.Tensor | None:
    return _SEGMENTS.get()


def segment_ids(doc_starts: np.ndarray, start: int, T: int) -> np.ndarray:
    """(T,) int64: document index, counted from 0 inside the window, of tokens ``start .. start+T-1``.

    ``doc_starts`` are the sorted start offsets of every document in the token stream (``*_docs.npy``).
    Example: documents start at 0, 5, 7; a window of 6 tokens from offset 3 covers documents
    0, 0, 1, 1, 2, 2 -> ``[0, 0, 1, 1, 2, 2]``.
    """
    pos = np.arange(start, start + T)
    doc = np.searchsorted(doc_starts, pos, side="right") - 1
    return (doc - doc[0]).astype(np.int64)


def segments_from_eot(x: torch.Tensor, eot_id: int) -> torch.Tensor:
    """Document ids from the end-of-text tokens of a packed batch x (B, T): a new document starts
    *after* every EOT, so the EOT itself belongs to the document it ends."""
    is_eot = (x == eot_id).long()
    return torch.cumsum(is_eot, dim=1) - is_eot


def docmask(q_pos: torch.Tensor, k_pos: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
    """(B, 1, Tq, Tk) bool mask: causal and same document. ``seg`` (B, T_total) by absolute position."""
    causal = k_pos[None, :] <= q_pos[:, None]                                  # (Tq, Tk)
    same = seg[:, q_pos][:, :, None] == seg[:, k_pos][:, None, :]              # (B, Tq, Tk)
    return (causal[None] & same)[:, None]


def useful_pair_fraction(seg: np.ndarray) -> float:
    """Fraction of the causal (query, key) pairs of a window that a document mask keeps.

    For segment lengths n_1..n_m in a window of T tokens: sum n_i (n_i + 1) / 2 over T (T + 1) / 2.
    One document -> 1.0; four equal documents -> about 0.25.
    """
    _, counts = np.unique(np.asarray(seg), return_counts=True)
    T = int(counts.sum())
    return float((counts * (counts + 1)).sum() / (T * (T + 1)))


@register("gqa-docmask")
class DocMaskGQAttention(GQAttention):
    """Baseline-0's GQA with an optional block-diagonal document mask (see the module docstring)."""

    def forward(self, x, positions, cache=None):
        seg = current_segments()
        if seg is None:
            return super().forward(x, positions, cache)
        B, T, _ = x.shape
        if seg.shape[0] != B or int(positions.max()) >= seg.shape[1]:
            raise ValueError(f"document segments {tuple(seg.shape)} do not cover batch {B} at positions up to "
                             f"{int(positions.max())}; set them with datax.packing.document_segments(seg)")
        q = self.q_norm(self.q_proj(x).view(B, T, self.H, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.KV, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.KV, self.hd).transpose(1, 2)
        cos, sin = self.rope(positions)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k_pos = positions
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat((cache["k"], k), dim=2), torch.cat((cache["v"], v), dim=2)
                k_pos = torch.cat((cache["pos"], positions))
            cache["k"], cache["v"], cache["pos"] = k, v, k_pos
        mask = docmask(positions, k_pos, seg.to(x.device))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=self.H != self.KV)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.hd))


@torch.no_grad()
def window_losses_docmask(model, data, n_windows: int = 256, T: int = 512, batch: int = 16, seed: int = 1234,
                          device="cpu", autocast_dtype=None) -> list[float]:
    """``frontierlab.evals.window_losses`` (same fixed windows, same order) with the document mask on.

    Use it to score a document-masked model the way it was trained; :func:`frontierlab.datax.packing.doc_losses`
    scores every model on single documents, where masked and unmasked attention see the same context.
    """
    was = model.training
    model.eval()
    starts, out = data.eval_windows(n_windows, T, seed), []
    for i in range(0, len(starts), batch):
        ss = starts[i:i + batch]
        x = torch.stack([data.window(s, T) for s in ss]).to(device)
        seg = torch.from_numpy(np.stack([segment_ids(data.doc_starts, s, T) for s in ss])).to(device)
        with document_segments(seg), torch.autocast(device_type=x.device.type, dtype=autocast_dtype,
                                                    enabled=autocast_dtype is not None):
            res = model(x, labels=x)
        out.extend(res.per_token_loss.float().mean(dim=1).tolist())
    model.train(was)
    return out


@torch.no_grad()
def doc_losses(model, data, max_docs: int = 200, T: int = 256, batch: int = 8, device="cpu") -> list[float]:
    """Mean next-token loss over the first ``T`` tokens of each of the first ``max_docs`` documents, every
    document scored on its own (right-padded, only its own tokens counted). A masked and an unmasked model
    see identical context here, so this is the fair per-document comparison of the two."""
    was = model.training
    model.eval()
    starts = data.doc_starts
    ends = np.append(starts[1:], len(data.tokens))
    items = [(int(s), int(min(e - s, T))) for s, e in zip(starts[:max_docs], ends[:max_docs]) if e - s >= 2]
    out = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        L = max(n for _, n in chunk)
        x = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, (s, n) in enumerate(chunk):
            x[j, :n] = data.window(s, n)
        x = x.to(device)
        tok = model(x, labels=x).per_token_loss                      # (b, L-1)
        for j, (_, n) in enumerate(chunk):
            out.append(float(tok[j, :n - 1].mean()))
    model.train(was)
    return out
