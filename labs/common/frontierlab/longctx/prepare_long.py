"""Long documents from FineWeb-Edu in the Data-v0 format (lessons 04.1 and 04.3).

    # free CPU: more held-out long documents for Eval v1 (validation and test only)
    python -m frontierlab.longctx.prepare_long --skip 20000 --docs 200000 --min-tokens 1024 \\
        --splits val test --out labs/common/data/v0-long

    # main path (Module 4 pilot): long training documents for the 8K and 32K stages, plus held-out ones
    python -m frontierlab.longctx.prepare_long --skip 2600000 --docs 9600000 --min-tokens 8192 \\
        --splits train val test --out labs/common/data/v0-long8k

What it does:

1. Streams FineWeb-Edu ``sample-10BT`` at Data-v0's pinned revision, skipping the first ``--skip``
   documents (the ones Data-v0 already used) and reading the next ``--docs``.
2. Normalises and hashes each document exactly like ``frontierlab.data.prepare`` and assigns its split
   **with the same hash rule**. A document that is in Data-v0's training split (or an exact duplicate
   of one) can therefore never land in a held-out split here: the split depends only on the text.
3. Tokenizes with Data-v0's tokenizer (``--tokenizer``; never retrains it, so ids mean the same thing)
   and keeps documents with at least ``--min-tokens`` tokens.
4. Writes ``{split}.bin``, ``{split}_docs.npy`` and ``meta.json`` (provenance, the rule, counts and
   SHA-256 of every file), readable by ``TokenData(split, root=--out)`` and ``LongDocData``.

FineWeb-Edu is web text: documents of 32K+ tokens are rare in it (about 0.04% of documents in the CPU
slice of Data-v0, holding about 1.7% of its tokens; measured with ``python -m frontierlab.longctx.data``).
Published long-context recipes add books and code repositories for that reason (lesson 04.3).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from frontierlab.data.prepare import (CONFIG, DATASET, DEFAULT_OUT, EOT, LICENSE, REVISION, doc_hash, sha256_file,
                                      split_of)


def stream(skip: int, n: int):
    from datasets import load_dataset
    ds = load_dataset(DATASET, name=CONFIG, split="train", streaming=True, revision=REVISION)
    for i, row in enumerate(ds.skip(skip)):
        if i == n:
            break
        yield row["text"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip", type=int, default=20000, help="documents to skip (Data-v0's --docs)")
    ap.add_argument("--docs", type=int, default=200000, help="documents to read after the skipped ones")
    ap.add_argument("--min-tokens", type=int, default=1024)
    ap.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_OUT / "tokenizer.json")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT.parent / "v0-long")
    ap.add_argument("--chunk", type=int, default=2000)
    a = ap.parse_args(argv)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(a.tokenizer))
    eot = tok.token_to_id(EOT)
    a.out.mkdir(parents=True, exist_ok=True)
    files = {s: open(a.out / f"{s}.bin", "wb") for s in a.splits}
    offsets = {s: [] for s in a.splits}
    n_tok = {s: 0 for s in a.splits}
    seen, read, dupes, short = set(), 0, 0, 0
    t0 = time.time()
    buf = {s: [] for s in a.splits}

    def flush(split):
        nonlocal short
        for enc in tok.encode_batch(buf[split]):
            if len(enc.ids) + 1 < a.min_tokens:
                short += 1
                continue
            ids = np.asarray(enc.ids + [eot], dtype=np.uint16)
            offsets[split].append(n_tok[split])
            n_tok[split] += ids.size
            ids.tofile(files[split])
        buf[split] = []

    for text in stream(a.skip, a.docs):
        read += 1
        h = doc_hash(text)
        if h in seen:
            dupes += 1
            continue
        seen.add(h)
        s = split_of(h)
        if s in buf:
            buf[s].append(text)
            if len(buf[s]) >= a.chunk:
                flush(s)
        if read % 50000 == 0:
            print(f"read {read:,} documents ({time.time() - t0:.0f}s)", flush=True)
    for s in a.splits:
        flush(s)
        files[s].close()
        np.save(a.out / f"{s}_docs.npy", np.asarray(offsets[s], dtype=np.int64))
    meta = {"name": f"Data-v0-long (>= {a.min_tokens} tokens)", "dataset": DATASET, "config": CONFIG,
            "revision": REVISION, "license": LICENSE, "skipped_documents": a.skip, "read_documents": read,
            "exact_duplicates_dropped": dupes, "shorter_than_min_dropped": short, "min_tokens": a.min_tokens,
            "vocab_size": tok.get_vocab_size(), "eot_id": eot, "tokenizer_sha256": sha256_file(a.tokenizer),
            "split_rule": "same as Data-v0: sha1(normalized text)[:8] mod 1000: <10 val, <20 test, else train"}
    for s in a.splits:
        meta[s] = {"documents": len(offsets[s]), "tokens": n_tok[s], "bin_sha256": sha256_file(a.out / f"{s}.bin"),
                   "docs_sha256": sha256_file(a.out / f"{s}_docs.npy")}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"done in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
