"""Prepare Data-v0: a fixed slice of FineWeb-Edu with provenance, split by document, tokenized.

    python -m frontierlab.data.prepare --docs 20000 --vocab 8192          # free CPU variant (~20M tokens)
    python -m frontierlab.data.prepare --docs 2600000 --vocab 32768       # main path (~2.5B tokens)

What it does, in order:

1. Streams the first ``--docs`` documents of FineWeb-Edu ``sample-10BT`` at a pinned revision.
2. Normalizes each document (Unicode NFC, collapsed whitespace) and hashes it (SHA-1). Exact
   duplicates are dropped. The split is chosen *from the hash* (``hash mod 1000``: < 10 -> val,
   < 20 -> test, else train), so identical documents always land in the same split — exact
   duplicates can never leak across splits, and the split of a document does not depend on how
   many documents were streamed. (Near-duplicates are Module 10's job.)
3. Trains a byte-level BPE tokenizer on (at most ``--tok-docs``) *train* documents only.
4. Writes, per split: ``{split}.bin`` (uint16 ids, documents separated by ``<|endoftext|>``),
   ``{split}_docs.npy`` (int64 start offset of every document, for document-level evaluation and
   document masking), and ``meta.json`` with provenance and SHA-256 of every file.

FineWeb-Edu is released under ODC-By 1.0 (https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np

DATASET = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"  # pinned 2026-10-02 (same as the MoE course)
LICENSE = "ODC-By 1.0"
EOT = "<|endoftext|>"
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data" / "v0"
SPLITS = ("train", "val", "test")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def doc_hash(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()


def split_of(h: str) -> str:
    b = int(h[:8], 16) % 1000
    return "val" if b < 10 else "test" if b < 20 else "train"


def stream_docs(n: int):
    from datasets import load_dataset
    ds = load_dataset(DATASET, name=CONFIG, split="train", streaming=True, revision=REVISION)
    for i, row in enumerate(ds):
        if i == n:
            break
        yield row["text"], row.get("id")


def train_tokenizer(docs, vocab: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(docs, trainer=trainer)
    return tok


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", type=int, default=20000, help="documents to stream (before dedup)")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--tok-docs", type=int, default=100000, help="max train documents for the tokenizer")
    ap.add_argument("--chunk", type=int, default=10000, help="documents tokenized per batch")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    if a.vocab > 65535:
        raise ValueError("uint16 token files need vocab <= 65535")
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Pass 1: stream, dedup, split. Documents are kept in memory per split as raw text; for the
    # main-path size (~2.6M docs, ~12 GB of text) run this on a machine with enough RAM, or lower --docs.
    seen, by_split, dupes = set(), {s: [] for s in SPLITS}, 0
    for text, _ in stream_docs(a.docs):
        h = doc_hash(text)
        if h in seen:
            dupes += 1
            continue
        seen.add(h)
        by_split[split_of(h)].append(text)
    print(f"streamed {a.docs} docs, dropped {dupes} exact duplicates, "
          + ", ".join(f"{s}={len(v)}" for s, v in by_split.items()) + f" ({time.time() - t0:.0f}s)")

    tok = train_tokenizer(by_split["train"][:a.tok_docs], a.vocab)
    tok.save(str(a.out / "tokenizer.json"))
    eot = tok.token_to_id(EOT)
    meta = {"name": "Data-v0", "dataset": DATASET, "config": CONFIG, "revision": REVISION, "license": LICENSE,
            "streamed_documents": a.docs, "exact_duplicates_dropped": dupes, "vocab_size": tok.get_vocab_size(),
            "eot_id": eot, "tokenizer_trained_on": f"first {min(a.tok_docs, len(by_split['train']))} train documents",
            "split_rule": "sha1(normalized text)[:8] mod 1000: <10 val, <20 test, else train",
            "tokenizer_sha256": sha256_file(a.out / "tokenizer.json")}

    # Pass 2: tokenize each split in chunks and append to its .bin file.
    for split in SPLITS:
        docs, offsets, n_tok = by_split[split], [], 0
        with open(a.out / f"{split}.bin", "wb") as f:
            for i in range(0, len(docs), a.chunk):
                for enc in tok.encode_batch(docs[i:i + a.chunk]):
                    ids = np.asarray(enc.ids + [eot], dtype=np.uint16)
                    offsets.append(n_tok)
                    n_tok += ids.size
                    ids.tofile(f)
        np.save(a.out / f"{split}_docs.npy", np.asarray(offsets, dtype=np.int64))
        meta[split] = {"documents": len(docs), "tokens": n_tok,
                       "bin_sha256": sha256_file(a.out / f"{split}.bin"),
                       "docs_sha256": sha256_file(a.out / f"{split}_docs.npy")}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"done in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
