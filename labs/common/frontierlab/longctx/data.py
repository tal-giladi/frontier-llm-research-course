"""Long-document data for context extension (lesson 04.3).

The training loop's ``TokenData.batch`` draws random windows from one long token stream in which
documents are separated by ``<|endoftext|>``. At a 256-token window that is harmless. At a long
window it means most of the "long context" a token sees belongs to *other, unrelated documents*:
Data-v0's median document is about 700 tokens, so a 2,048-token window usually spans three or more
documents. The model is then trained to use long context that carries no information, and the run
says nothing about whether it can use a long document.

This module provides:

* :func:`doc_lengths` and :func:`same_doc_context` — measure the problem on any Data-v0 split: how much
  same-document context the tokens of a random window actually have.
* :class:`LongDocData` — a drop-in for ``TokenData`` whose ``batch`` draws windows that lie entirely
  inside one document of at least ``T`` tokens (uniform over all such windows), mixed with ordinary
  windows at a chosen fraction so short-context data stays in the mix. It uses the caller's
  ``torch.Generator``, so exact resume (lesson 01.1) still holds.
* :func:`doc_start_windows` — the first ``T`` tokens of every document of length >= ``T``: evaluation
  windows in which "position" means position in the document.
* ``python -m frontierlab.longctx.data`` — prints the long-document statistics of a split.

Document masking (an attention mask that stops tokens from attending across a document boundary
inside one packed window; Llama 3 report, section 3.2) is the other honest option. It needs a
per-sequence mask, which the course attention interface (one ``positions`` vector per batch) does not
carry, so the course uses within-document windows instead; lesson 04.3 discusses the difference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from frontierlab.data.loader import TokenData
from frontierlab.data.prepare import DEFAULT_OUT


def doc_lengths(data: TokenData) -> np.ndarray:
    """Length in tokens (including the trailing end-of-text token) of every document, in file order."""
    if data.doc_starts is None:
        raise ValueError(f"{data.split}_docs.npy is missing; re-run frontierlab.data.prepare")
    return np.diff(np.append(data.doc_starts, len(data.tokens))).astype(np.int64)


def same_doc_context(data: TokenData, T: int, n_windows: int = 512, seed: int = 0) -> np.ndarray:
    """For random windows of length ``T``: (n_windows, T) same-document context of every token.

    Entry [w, p] is how many earlier tokens of the window belong to the same document as token p
    (0 for the first token of a document). With within-document windows it would be exactly ``p``.
    """
    starts = data.doc_starts
    g = torch.Generator().manual_seed(seed)
    ws = torch.randint(0, len(data.tokens) - T, (n_windows,), generator=g).numpy()
    pos = ws[:, None] + np.arange(T)[None, :]                                # absolute token index
    doc_start = starts[np.searchsorted(starts, pos, side="right") - 1]      # start of each token's document
    return np.minimum(pos - doc_start, np.arange(T)[None, :])


class LongDocData(TokenData):
    """``TokenData`` whose ``batch(B, T, ...)`` draws within-document windows from long documents.

    ``long_fraction`` of the sequences in a batch (decided per sequence with the same generator) are
    within-document windows of documents with at least ``T`` tokens; the rest are ordinary random
    windows. ``long_fraction=1`` is "long documents only"; lower values keep short-context data in
    the mix, which is what published recipes do to protect short-context quality (lesson 04.3).
    With ``short_root`` the ordinary windows come from another prepared folder (for example long
    documents from ``frontierlab.longctx.prepare_long`` in ``root`` and Data-v0 in ``short_root``).
    """

    def __init__(self, split: str = "train", root: str | Path = DEFAULT_OUT, long_fraction: float = 1.0,
                 short_root: str | Path | None = None):
        super().__init__(split, root)
        if not 0.0 <= long_fraction <= 1.0:
            raise ValueError("long_fraction must be in [0, 1]")
        self.long_fraction = long_fraction
        # ordinary windows come from ``short_root`` if given (e.g. long documents in ``root``, Data-v0 here)
        self.short = TokenData(split, short_root) if short_root is not None else self
        self.lengths = doc_lengths(self)
        self._index: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}

    def _windows(self, T: int):
        """Docs with >= T tokens, the cumulative count of valid window starts in them, and the total."""
        if T not in self._index:
            ok = np.nonzero(self.lengths >= T)[0]
            if ok.size == 0:
                raise ValueError(f"no document in {self.split} has {T} tokens; prepare long documents first")
            counts = self.lengths[ok] - T + 1
            self._index[T] = (ok, np.cumsum(counts), int(counts.sum()))
        return self._index[T]

    def stats(self, T: int) -> dict:
        ok, _, total = self._windows(T)
        return {"T": T, "documents": int(ok.size), "tokens_in_them": int(self.lengths[ok].sum()),
                "valid_window_starts": total, "disjoint_windows": int((self.lengths[ok] // T).sum())}

    def within_doc_start(self, T: int, generator: torch.Generator) -> int:
        ok, cum, total = self._windows(T)
        u = int(torch.randint(0, total, (1,), generator=generator))
        j = int(np.searchsorted(cum, u, side="right"))
        before = int(cum[j - 1]) if j else 0
        return int(self.doc_starts[ok[j]]) + (u - before)

    def batch(self, B: int, T: int, generator: torch.Generator, device="cpu") -> torch.Tensor:
        rows = []
        for _ in range(B):
            long = self.long_fraction >= 1.0 or float(torch.rand((), generator=generator)) < self.long_fraction
            if long:
                rows.append(self.window(self.within_doc_start(T, generator), T))
            else:
                s = int(torch.randint(0, len(self.short.tokens) - T, (1,), generator=generator))
                rows.append(self.short.window(s, T))
        return torch.stack(rows).to(device)


def doc_start_windows(data: TokenData, T: int, max_docs: int | None = None) -> list[int]:
    """Start offsets of every document with at least ``T`` tokens (file order, first ``max_docs``)."""
    lengths = doc_lengths(data)
    starts = [int(s) for s, n in zip(data.doc_starts, lengths) if n >= T]
    return starts[:max_docs] if max_docs else starts


def main(argv=None):
    ap = argparse.ArgumentParser(description="Long-document statistics of a Data-v0 split")
    ap.add_argument("--split", default="train")
    ap.add_argument("--data", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--lengths", type=int, nargs="+", default=[256, 1024, 2048, 4096, 8192, 32768])
    a = ap.parse_args(argv)
    d = TokenData(a.split, a.data)
    L = doc_lengths(d)
    print(f"{a.split}: {L.size} documents, {L.sum():,} tokens, median {np.median(L):.0f}, "
          f"p90 {np.percentile(L, 90):.0f}, p99 {np.percentile(L, 99):.0f}, max {L.max():,}")
    print(f"{'T':>7} {'docs >= T':>10} {'% docs':>7} {'% tokens':>9} {'disjoint T-windows':>19} "
          f"{'random T-window: tokens with >= T/2 same-doc context':>54}")
    for T in a.lengths:
        m = L >= T
        ctx = same_doc_context(d, T, n_windows=256) if len(d.tokens) > T else None
        frac = float((ctx >= T // 2).mean()) if ctx is not None else float("nan")
        print(f"{T:>7} {int(m.sum()):>10} {100 * m.mean():>6.2f}% {100 * L[m].sum() / L.sum():>8.2f}% "
              f"{int((L[m] // T).sum()):>19} {100 * frac:>53.1f}%")


if __name__ == "__main__":
    main()
