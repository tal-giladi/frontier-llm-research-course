"""Token-compressed KV entries in the style of DeepSeek-V4's CSA and HCA (lesson 05.4, extension).

What DeepSeek-V4's report documents (arXiv 2606.19348, section 2.3; checked 2026-10-04), in our words:

* **CSA (Compressed Sparse Attention)** compresses "the KV cache of every m tokens into one entry". Each
  entry is a weighted sum over 2m tokens from two paths: path a over its own block of m tokens and path b
  over the previous block, with softmax weights normalised over all 2m terms, so consecutive entries
  overlap. A lightning indexer (DSA, lesson 05.2) selects the top-k compressed entries per query, and an
  additional sliding-window branch keeps the most recent n_win tokens uncompressed.
* **HCA (Heavily Compressed Attention)** compresses every m' tokens (m' >> m) into one entry, without
  overlap, and attends densely over all compressed entries (no selection).
* both use shared-KV MQA (each compressed entry is key and value), RMSNorm on queries and on the KV entry,
  partial RoPE on the last 64 dimensions, a learned sink logit per head, a grouped output projection, and
  the layers are interleaved; KV is stored in FP8 except the RoPE dimensions (BF16); the indexer's
  attention runs in FP4.

Not stated in section 2.3 (and therefore not used as facts anywhere in the course): the numeric values of
m, m', k, n_win, the indexer head count and width, the attention head count and width, and the exact
CSA/HCA layer pattern. Everything below takes them as arguments.

What this module implements (a didactic reference, not V4's code):

* :func:`compress_hca` — non-overlapping blocks: entry_i = sum_{j in block i} softmax_j(z_j) * c_j, per
  channel, over the m' tokens of block i.
* :func:`compress_csa` — overlapping blocks: entry_i = sum_{j in block i} S^a_j * c^a_j + sum_{j in block
  i-1} S^b_j * c^b_j, with S the softmax over the 2m logits (z^a over block i, z^b over block i-1),
  per channel. For i = 0 the previous block does not exist and only path a is used.
* :func:`usable_entries` — causality: a compressed entry exists only once its whole block has been seen,
  so a query at position t may use entries of complete blocks only, (t + 1) // m of them. The tokens of
  the incomplete block are what the sliding window covers.
* :func:`kv_entries` / :func:`attended_entries` — the cost model of lesson 05.4.
"""

from __future__ import annotations

import torch


def _blocks(x: torch.Tensor, m: int) -> torch.Tensor:
    """(B, S, d) -> (B, S // m, m, d); an incomplete final block is dropped (it has no entry yet)."""
    B, S, d = x.shape
    n = S // m
    return x[:, : n * m].reshape(B, n, m, d)


def compress_hca(c: torch.Tensor, z: torch.Tensor, m: int) -> torch.Tensor:
    """c, z (B, S, d) -> (B, S // m, d): softmax(z) over each block of m tokens, per channel, weights c."""
    cb, zb = _blocks(c, m), _blocks(z, m)
    return (torch.softmax(zb, dim=2) * cb).sum(dim=2)


def compress_csa(ca: torch.Tensor, za: torch.Tensor, cb: torch.Tensor, zb: torch.Tensor, m: int) -> torch.Tensor:
    """Overlapping compression over 2m tokens per entry (path a: own block, path b: previous block).

    ca, za, cb, zb (B, S, d) -> (B, S // m, d). The softmax normalises over the 2m logits of an entry (the
    m of za in block i and the m of zb in block i - 1), per channel.
    """
    A_c, A_z = _blocks(ca, m), _blocks(za, m)                         # (B, n, m, d)
    B_c, B_z = _blocks(cb, m), _blocks(zb, m)
    n = A_c.shape[1]
    if n == 0:
        return ca.new_zeros(ca.shape[0], 0, ca.shape[2])
    pad_z = torch.full_like(B_z[:, :1], float("-inf"))               # block -1 does not exist
    pad_c = torch.zeros_like(B_c[:, :1])
    prev_z = torch.cat((pad_z, B_z[:, : n - 1]), dim=1)
    prev_c = torch.cat((pad_c, B_c[:, : n - 1]), dim=1)
    logits = torch.cat((A_z, prev_z), dim=2)                         # (B, n, 2m, d)
    vals = torch.cat((A_c, prev_c), dim=2)
    return (torch.softmax(logits, dim=2) * vals).sum(dim=2)


def usable_entries(t: int, m: int) -> int:
    """Compressed entries a query at position t (0-based) may attend to: complete blocks only."""
    return (t + 1) // m


def kv_entries(S: int, *, m: int | None = None, window: int = 0) -> int:
    """Cached KV entries of one layer after S tokens: S (dense), S // m (+ the window's tokens if any)."""
    if m is None:
        return S
    return S // m + min(S, window)


def attended_entries(S: int, *, kind: str, m: int | None = None, k: int | None = None, window: int = 0) -> int:
    """Entries one decode query attends to at context S.

    dense: S;  hca: S // m;  csa: min(k, S // m) selected compressed entries + min(S, window) recent tokens.
    """
    if kind == "dense":
        return S
    if kind == "hca":
        return S // m
    if kind == "csa":
        return min(k, S // m) + min(S, window)
    raise ValueError(kind)
