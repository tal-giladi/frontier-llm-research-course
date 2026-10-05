"""Extra training sources for Module 10, in the Data-v0 format, with provenance.

    python -m frontierlab.datax.sources list
    python -m frontierlab.datax.sources prepare web --docs 20000          # FineWeb (unfiltered web)
    python -m frontierlab.datax.sources prepare wiki --docs 4000          # English Wikipedia
    python -m frontierlab.datax.sources prepare math --docs 8000          # FineMath-4+
    python -m frontierlab.datax.sources prepare fw2-fra --docs 2000       # FineWeb2, French (lesson 10.6)

Every source is a pinned Hugging Face dataset revision (:data:`SOURCES`). ``prepare`` does what
``frontierlab.data.prepare`` does for Data-v0, with three differences:

* it never trains a tokenizer: every source is tokenized with **Data-v0's tokenizer**, so token ids
  mean the same thing in every source and a mixture of sources is a valid training stream;
* the split of every document comes from **Data-v0's hash rule** (``split_of(doc_hash(text))``), so a
  document that is in Data-v0's validation or test split, or an exact copy of one in any source,
  lands in the same held-out split here and can never become a training document;
* it writes per-document provenance (``{split}_provenance.jsonl``: dataset id, URL, SHA-1 of the
  normalized text) and a ``provenance`` block in ``meta.json`` (dataset, config, revision, licence,
  licence URL, attribution and share-alike flags, the exact command), which
  :mod:`frontierlab.datax.provenance` checks.

Output: ``labs/common/data/m10/<name>/`` (gitignored). Downloads are streamed; only the rows read are
fetched (plus parquet row-group overhead). Sizes and times measured in the build are in the lessons.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from frontierlab.data.prepare import DEFAULT_OUT, EOT, SPLITS, doc_hash, sha256_file, split_of

M10_ROOT = DEFAULT_OUT.parent / "m10"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset: str
    config: str | None
    revision: str                  # a 40-hex commit of the dataset repository: never a branch name
    license: str
    license_url: str
    text_field: str = "text"
    id_field: str | None = "id"
    url_field: str | None = "url"
    attribution_required: bool = True
    share_alike: bool = False
    notes: str = ""
    extra_fields: tuple = field(default_factory=tuple)   # kept in provenance rows (e.g. a quality score)


SOURCES: dict[str, SourceSpec] = {s.name: s for s in [
    SourceSpec("edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
               "ODC-By 1.0", "https://opendatacommons.org/licenses/by/1-0/",
               notes="Data-v0 itself; prepared by frontierlab.data.prepare. Listed here for the manifest."),
    SourceSpec("web", "HuggingFaceFW/fineweb", "sample-10BT", "9bb295ddab0e05d785b879661af7260fed5140fc",
               "ODC-By 1.0", "https://opendatacommons.org/licenses/by/1-0/",
               notes="Unfiltered (heuristically filtered, not model-filtered) web text; FineWeb-Edu is a subset of FineWeb.",
               extra_fields=("dump", "language_score")),
    SourceSpec("wiki", "wikimedia/wikipedia", "20231101.en", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
               "CC BY-SA 3.0 and GFDL", "https://creativecommons.org/licenses/by-sa/3.0/", share_alike=True,
               notes="Share-alike: redistributing the text or derived text datasets carries the licence with it.",
               extra_fields=("title",)),
    SourceSpec("math", "HuggingFaceTB/finemath", "finemath-4plus", "e92b25a616738fe95dc186b64dfb19f9c8525594",
               "ODC-By 1.0", "https://opendatacommons.org/licenses/by/1-0/", id_field=None,
               notes="Math pages from Common Crawl, classifier score >= 4.", extra_fields=("score",)),
    SourceSpec("annot", "HuggingFaceFW/fineweb-edu-llama3-annotations", None, "72df4c92fb1b48beceb16016e8f695ec40a6c3a5",
               "ODC-By 1.0", "https://opendatacommons.org/licenses/by/1-0/", id_field=None, url_field=None,
               notes="Educational scores 0-5 written by Llama-3-70B-Instruct (FineWeb paper section 4); check the "
                     "Llama 3 licence's conditions on model outputs before redistributing a model trained on them.",
               extra_fields=("score",)),
    SourceSpec("fw2-fra", "HuggingFaceFW/fineweb-2", "fra_Latn", "af9c13333eb981300149d5ca60a8e9d659b276b9",
               "ODC-By 1.0", "https://opendatacommons.org/licenses/by/1-0/", extra_fields=("language_score", "minhash_cluster_size")),
    SourceSpec("fw2-deu", "HuggingFaceFW/fineweb-2", "deu_Latn", "af9c13333eb981300149d5ca60a8e9d659b276b9",
               "ODC-By 1.0", "https://opendatacommons.org/licenses/by/1-0/", extra_fields=("language_score", "minhash_cluster_size")),
    SourceSpec("fw2-heb", "HuggingFaceFW/fineweb-2", "heb_Hebr", "af9c13333eb981300149d5ca60a8e9d659b276b9",
               "ODC-By 1.0", "https://opendatacommons.org/licenses/by/1-0/", extra_fields=("language_score", "minhash_cluster_size")),
]}


def spec_dict(spec: SourceSpec) -> dict:
    d = asdict(spec)
    d["extra_fields"] = list(spec.extra_fields)
    return d


def stream_rows(spec: SourceSpec, n: int, skip: int = 0):
    """The first ``n`` rows (after ``skip``) of the pinned dataset, streamed."""
    from datasets import load_dataset
    ds = load_dataset(spec.dataset, name=spec.config, split="train", streaming=True, revision=spec.revision)
    if skip:
        ds = ds.skip(skip)
    for i, row in enumerate(ds):
        if i == n:
            break
        yield row


def load_tokenizer(path: str | Path = DEFAULT_OUT / "tokenizer.json"):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


def write_source(out: Path, docs: dict, prov: dict, meta: dict, tok, chunk: int = 2000) -> dict:
    """Tokenize ``docs[split]`` (lists of text) with ``tok`` and write the Data-v0 file layout.

    ``prov[split]`` is a list of per-document dicts written to ``{split}_provenance.jsonl`` in the
    same order (its ``tokens`` field is filled in here). Returns the completed ``meta``.
    """
    out.mkdir(parents=True, exist_ok=True)
    eot = tok.token_to_id(EOT)
    for split in SPLITS:
        texts, rows, offsets, n_tok = docs.get(split, []), prov.get(split, []), [], 0
        with open(out / f"{split}.bin", "wb") as f:
            for i in range(0, len(texts), chunk):
                for j, enc in enumerate(tok.encode_batch(texts[i:i + chunk])):
                    ids = np.asarray(enc.ids + [eot], dtype=np.uint16)
                    offsets.append(n_tok)
                    rows[i + j]["tokens"] = int(ids.size)
                    n_tok += ids.size
                    ids.tofile(f)
        np.save(out / f"{split}_docs.npy", np.asarray(offsets, dtype=np.int64))
        with open(out / f"{split}_provenance.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        meta[split] = {"documents": len(texts), "tokens": int(n_tok),
                       "bin_sha256": sha256_file(out / f"{split}.bin"),
                       "docs_sha256": sha256_file(out / f"{split}_docs.npy"),
                       "provenance_sha256": sha256_file(out / f"{split}_provenance.jsonl")}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def prepare(name: str, n_docs: int, out: Path | None = None, skip: int = 0, min_chars: int = 200,
            tokenizer: Path = DEFAULT_OUT / "tokenizer.json", command: str = "") -> dict:
    spec = SOURCES[name]
    if name == "edu":
        raise SystemExit("edu is Data-v0: python -m frontierlab.data.prepare")
    out = out or M10_ROOT / name
    tok = load_tokenizer(tokenizer)
    t0 = time.time()
    seen, dupes, short = set(), 0, 0
    docs = {s: [] for s in SPLITS}
    prov = {s: [] for s in SPLITS}
    for row in stream_rows(spec, n_docs, skip):
        text = row[spec.text_field]
        if len(text) < min_chars:
            short += 1
            continue
        h = doc_hash(text)
        if h in seen:
            dupes += 1
            continue
        seen.add(h)
        s = split_of(h)
        docs[s].append(text)
        p = {"sha1": h}
        if spec.id_field and spec.id_field in row:
            p["id"] = str(row[spec.id_field])
        if spec.url_field and spec.url_field in row:
            p["url"] = row[spec.url_field]
        for k in spec.extra_fields:
            if k in row:
                p[k] = row[k]
        prov[s].append(p)
    meta = {"name": f"m10-{name}", "dataset": spec.dataset, "config": spec.config, "revision": spec.revision,
            "license": spec.license, "vocab_size": tok.get_vocab_size(), "eot_id": tok.token_to_id(EOT),
            "tokenizer_sha256": sha256_file(tokenizer), "streamed_documents": n_docs, "skipped_documents": skip,
            "exact_duplicates_dropped": dupes, "shorter_than_min_chars_dropped": short, "min_chars": min_chars,
            "split_rule": "same as Data-v0: sha1(normalized text)[:8] mod 1000: <10 val, <20 test, else train",
            "provenance": {**spec_dict(spec), "command": command or " ".join(sys.argv),
                           "prepared_unix_time": int(time.time())}}
    meta = write_source(out, docs, prov, meta, tok)
    meta["seconds"] = round(time.time() - t0, 1)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("prepare")
    p.add_argument("name", choices=sorted(SOURCES))
    p.add_argument("--docs", type=int, default=20000)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_OUT / "tokenizer.json")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for s in SOURCES.values():
            print(f"{s.name:8s} {s.dataset}{':' + s.config if s.config else ''}@{s.revision[:10]}  {s.license}"
                  + ("  [share-alike]" if s.share_alike else ""))
        return
    meta = prepare(a.name, a.docs, a.out, a.skip, a.min_chars, a.tokenizer,
                   command=f"python -m frontierlab.datax.sources prepare {a.name} --docs {a.docs}"
                           + (f" --skip {a.skip}" if a.skip else ""))
    print(json.dumps({k: v for k, v in meta.items() if k != "provenance"}, indent=2))


if __name__ == "__main__":
    main()
