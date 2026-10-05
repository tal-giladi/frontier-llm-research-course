"""Deterministic, resumable mixture sampling with exact token accounting per source (lesson 10.1).

A mixture is a list of sources (prepared Data-v0-format folders) with token weights. The sampler
turns it into an infinite stream of packed windows of ``T`` tokens. Everything about window ``k`` is a
pure function of ``k`` and the mixture spec, so:

* **resume** is ``seek(k)``: no generator state to save, nothing to replay (``k`` = windows already
  consumed = step × grad_accum × batch);
* **token accounting is exact and known in advance**: :meth:`MixtureSampler.accounting` gives, for any
  ``k``, the tokens, windows, epochs and documents started per source;
* two runs with the same spec see bit-identical windows (``tests/test_datax.py`` checks a stopped and
  resumed stream against an uninterrupted one).

How window ``k`` is chosen (all integers; :class:`MixtureSpec` holds the inputs):

1. **Block schedule.** Windows come in blocks of ``P`` (``block``). The token weights are turned into
   integer window counts ``n_s`` per block by largest remainder (``sum n_s = P``), so after every
   complete block source ``s`` has supplied exactly ``n_s · T`` tokens per block. The order of the
   ``P`` slots inside block ``b`` is a permutation drawn from ``seed`` and ``b``. Window ``k`` is slot
   ``j = k mod P`` of block ``b = k // P``; its source is ``order_b[j]`` and its index inside that
   source's own stream is ``b · n_s`` + (how many earlier slots of block ``b`` belong to ``s``).
2. **Per-source stream.** Each source is an infinite stream of epochs. Epoch ``e`` is the source's
   documents in a permutation drawn from ``seed``, the source index and ``e``, concatenated (each
   document ends with ``<|endoftext|>``). Source window ``w`` is tokens ``[w·T, (w+1)·T)`` of that
   stream; a document can be split across two windows, and a window can cross an epoch boundary.
3. **Segments.** With the window, the sampler returns the document index of every position (counted
   from 0 in the window), for document masking (:mod:`frontierlab.datax.packing`).

The realised token shares differ from the requested weights only by the rounding to ``n_s / P``
(reported as ``realised_weights``). An epoch count above about 4 is flagged: Muennighoff et al.
(2023, abstract) find that up to 4 epochs of repetition is close to fresh data, with diminishing
returns beyond.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from frontierlab.data.loader import TokenData
from frontierlab.datax.packing import document_segments

REPEAT_WARN_EPOCHS = 4.0


@dataclass
class SourceRef:
    name: str
    root: str                 # prepared folder (Data-v0 layout)
    weight: float             # token share before normalisation
    split: str = "train"


@dataclass
class MixtureSpec:
    sources: list[SourceRef]
    seed: int = 0
    block: int = 100          # windows per block (weights are rounded to multiples of 1/block)
    name: str = "mixture"
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "MixtureSpec":
        d = dict(d)
        d["sources"] = [SourceRef(**s) for s in d["sources"]]
        return cls(**d)

    @classmethod
    def load(cls, path: str | Path) -> "MixtureSpec":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


def window_counts(weights, block: int) -> np.ndarray:
    """Integer windows per block for each source by largest remainder: sum = block, zero weight -> zero.

    Example: weights (0.5, 0.3, 0.2), block 7 -> exact (3.5, 2.1, 1.4) -> floors (3, 2, 1) leave one
    window, which goes to the largest remainder (0.5): (4, 2, 1).
    """
    w = np.asarray(weights, dtype=np.float64)
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    exact = w / w.sum() * block
    n = np.floor(exact).astype(np.int64)
    rest = block - int(n.sum())
    order = np.argsort(-(exact - n), kind="stable")
    n[order[:rest]] += 1
    return n


def _rng(*key) -> np.random.Generator:
    """A NumPy generator seeded from a tuple of integers (stable across runs and platforms)."""
    return np.random.default_rng(np.random.SeedSequence([int(k) & 0xFFFFFFFF for k in key]))


class _Source:
    """One source's infinite, epoch-shuffled stream of documents."""

    def __init__(self, ref: SourceRef, index: int, seed: int, cache: int = 4):
        self.ref, self.index, self.seed = ref, index, seed
        self.data = TokenData(ref.split, ref.root)
        if self.data.doc_starts is None:
            raise ValueError(f"{ref.root}/{ref.split}_docs.npy missing")
        self.starts = self.data.doc_starts.astype(np.int64)
        self.lengths = np.diff(np.append(self.starts, len(self.data.tokens))).astype(np.int64)
        self.total = int(self.lengths.sum())          # tokens per epoch
        self._epochs: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._cache = cache

    def epoch(self, e: int):
        """(document order, cumulative token ends in that order) of epoch e."""
        if e not in self._epochs:
            perm = _rng(self.seed, 1000 + self.index, e).permutation(self.lengths.size)
            self._epochs[e] = (perm, np.cumsum(self.lengths[perm]))
            if len(self._epochs) > self._cache:
                self._epochs.popitem(last=False)
        return self._epochs[e]

    def window(self, w: int, T: int) -> tuple[np.ndarray, np.ndarray]:
        """Tokens (T,) uint16 and segment ids (T,) of source window ``w``."""
        pos, end = w * T, (w + 1) * T
        toks, segs, seg = [], [], 0
        while pos < end:
            e, off = divmod(pos, self.total)
            perm, cum = self.epoch(e)
            i = int(np.searchsorted(cum, off, side="right"))
            doc_end = int(cum[i])
            take = min(end - pos, doc_end - off)
            doc_off = off - (doc_end - int(self.lengths[perm[i]]))
            s = int(self.starts[perm[i]]) + doc_off
            toks.append(np.asarray(self.data.tokens[s:s + take]))
            segs.append(np.full(take, seg, dtype=np.int64))
            seg += 1
            pos += take
        return np.concatenate(toks), np.concatenate(segs)


class MixtureSampler:
    """Packed windows from a :class:`MixtureSpec`; a drop-in for ``TokenData`` in the training loop.

    ``batch(B, T, generator, device)`` ignores ``generator`` (the stream is a function of the window
    counter) and, if ``doc_mask`` is set, leaves the batch's segment ids in
    :func:`frontierlab.datax.packing.document_segments` for the forward pass that follows.
    """

    def __init__(self, spec: MixtureSpec, doc_mask: bool = False):
        self.spec = spec
        self.sources = [_Source(r, i, spec.seed) for i, r in enumerate(spec.sources)]
        self.counts = window_counts([r.weight for r in spec.sources], spec.block)
        self.meta = dict(self.sources[0].data.meta)
        vocab = {s.data.meta.get("vocab_size") for s in self.sources}
        tok = {s.data.meta.get("tokenizer_sha256") for s in self.sources}
        if len(vocab) > 1 or len(tok) > 1:
            raise ValueError(f"sources use different tokenizers or vocabularies: {vocab} {tok}")
        self.tokens = self.sources[0].data.tokens      # for code that only reads .tokens (len)
        self.doc_mask = doc_mask
        self.k = 0                                      # windows consumed
        self._blocks: OrderedDict[int, np.ndarray] = OrderedDict()
        self._segments_cm = None

    # ------------------------------------------------------------------ schedule
    def block_order(self, b: int) -> np.ndarray:
        if b not in self._blocks:
            slots = np.repeat(np.arange(len(self.sources)), self.counts)
            self._blocks[b] = slots[_rng(self.spec.seed, 7, b).permutation(slots.size)]
            if len(self._blocks) > 8:
                self._blocks.popitem(last=False)
        return self._blocks[b]

    def locate(self, k: int) -> tuple[int, int]:
        """(source index, window index inside that source) of global window ``k``."""
        b, j = divmod(k, self.spec.block)
        order = self.block_order(b)
        s = int(order[j])
        return s, b * int(self.counts[s]) + int((order[:j] == s).sum())

    def window(self, k: int, T: int) -> tuple[np.ndarray, np.ndarray, int]:
        s, w = self.locate(k)
        toks, seg = self.sources[s].window(w, T)
        return toks, seg, s

    # ------------------------------------------------------------------ loop interface
    def seek(self, k: int) -> None:
        self.k = int(k)

    def state_dict(self) -> dict:
        return {"windows": self.k, "spec_digest": self.spec.digest()}

    def load_state_dict(self, sd: dict) -> None:
        if sd.get("spec_digest") != self.spec.digest():
            raise ValueError("mixture state belongs to a different mixture spec")
        self.seek(sd["windows"])

    def next_windows(self, B: int, T: int):
        rows, segs, srcs = [], [], []
        for _ in range(B):
            t, s, src = self.window(self.k, T)
            rows.append(t.astype(np.int64))
            segs.append(s)
            srcs.append(src)
            self.k += 1
        return torch.from_numpy(np.stack(rows)), torch.from_numpy(np.stack(segs)), srcs

    def batch(self, B: int, T: int, generator=None, device="cpu") -> torch.Tensor:
        x, seg, _ = self.next_windows(B, T)
        if self.doc_mask:
            # replace any previous batch's segments; the loop's forward runs right after this call
            if self._segments_cm is not None:
                self._segments_cm.__exit__(None, None, None)
            self._segments_cm = document_segments(seg.to(device))
            self._segments_cm.__enter__()
        return x.to(device)

    def close(self) -> None:
        if self._segments_cm is not None:
            self._segments_cm.__exit__(None, None, None)
            self._segments_cm = None

    # ------------------------------------------------------------------ accounting
    def windows_per_source(self, k: int) -> np.ndarray:
        """Exact windows supplied by each source in global windows [0, k)."""
        b, j = divmod(int(k), self.spec.block)
        out = self.counts * b
        if j:
            out = out + np.bincount(self.block_order(b)[:j], minlength=len(self.sources))
        return out.astype(np.int64)

    def accounting(self, k: int, T: int) -> dict:
        """Per source: windows, tokens, share, epochs, documents started; plus the realised weights."""
        wins = self.windows_per_source(k)
        out = {"windows": int(k), "T": T, "tokens": int(k) * T, "sources": {}}
        for s, src in enumerate(self.sources):
            tok = int(wins[s]) * T
            ep = tok / src.total
            full, rem = divmod(tok, src.total)
            started = full * src.lengths.size
            if rem:
                perm, cum = src.epoch(full)
                started += int(np.searchsorted(cum, rem - 1, side="right")) + 1
            out["sources"][src.ref.name] = {
                "root": src.ref.root, "windows": int(wins[s]), "tokens": tok,
                "share": tok / max(1, int(k) * T), "epochs": ep, "documents_started": int(started),
                "unique_tokens": src.total, "repeat_warning": ep > REPEAT_WARN_EPOCHS}
        out["requested_weights"] = {r.name: r.weight / sum(x.weight for x in self.spec.sources) for r in self.spec.sources}
        out["realised_weights"] = {r.name: int(c) / self.spec.block for r, c in zip(self.spec.sources, self.counts)}
        out["spec_digest"] = self.spec.digest()
        return out


def make_source_subset(src_root: str | Path, out: str | Path, doc_indices, split: str = "train",
                       note: str = "") -> dict:
    """Write a new source folder holding only documents ``doc_indices`` (in that order) of ``src_root``.

    Used for filtered subsets (lesson 10.2: classifier top-k%). The meta keeps the parent's provenance
    and records the selection (count and SHA-256 of the index list), so the subset is traceable.
    """
    from frontierlab.data.prepare import sha256_file
    src, out = Path(src_root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    d = TokenData(split, src)
    idx = np.asarray(doc_indices, dtype=np.int64)
    ends = np.append(d.doc_starts[1:], len(d.tokens))
    offsets, n = [], 0
    with open(out / f"{split}.bin", "wb") as f:
        for i in idx:
            a, b = int(d.doc_starts[i]), int(ends[i])
            offsets.append(n)
            n += b - a
            np.asarray(d.tokens[a:b]).tofile(f)
    np.save(out / f"{split}_docs.npy", np.asarray(offsets, dtype=np.int64))
    meta = {k: v for k, v in d.meta.items() if k not in ("train", "val", "test")}
    meta.update(name=f"{d.meta.get('name', src.name)}-subset", parent=str(src), selection={
        "split": split, "documents": int(idx.size), "of": int(d.doc_starts.size),
        "indices_sha256": hashlib.sha256(idx.tobytes()).hexdigest(), "note": note})
    meta[split] = {"documents": int(idx.size), "tokens": int(n), "bin_sha256": sha256_file(out / f"{split}.bin"),
                   "docs_sha256": sha256_file(out / f"{split}_docs.npy")}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta
