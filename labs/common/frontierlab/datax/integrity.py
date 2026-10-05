"""The data-integrity report of a training mixture (lesson 10.1, Module 10 project).

    python -m frontierlab.datax.integrity --train edu=labs/common/data/v0 web=labs/common/data/m10/web \\
        --heldout labs/common/data/v0 --lambada --eval-v1 labs/common/data/v0-long --out runs/m10/integrity.json

Four sections, each with the definition it used stored next to the numbers:

1. **Provenance** — :func:`frontierlab.datax.provenance.check_source` on every training source.
2. **Exact duplicates across splits** — SHA-1 of the normalized text of every document of every
   training source against every held-out document (Data-v0's ``val`` and ``test``). With Data-v0's
   hash split the count must be 0 for sources prepared by the course; a source prepared another way
   (or a held-out split used as a source) shows up here.
3. **Near-duplicates across splits** — MinHash/LSH (:mod:`frontierlab.datax.neardup`), word 5-gram
   shingles, verified pairs with exact Jaccard >= ``--jaccard`` between any training document and any
   held-out document.
4. **Benchmark leakage** — word 13-gram overlap (:mod:`frontierlab.datax.leakage`) of Eval v0's
   held-out windows, the LAMBADA passages and Eval v1's haystack documents with the training index,
   plus a **planted positive control**: the same check on a copy of the training texts with
   ``--plant`` LAMBADA passages inserted must flag every planted passage.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from frontierlab.data.loader import TokenData
from frontierlab.data.prepare import DEFAULT_OUT, doc_hash
from frontierlab.datax import leakage, neardup, provenance
from frontierlab.datax.sources import load_tokenizer


def texts_of(root, split: str, tok, max_docs: int | None = None) -> list[str]:
    return neardup.decode_docs(TokenData(split, root), tok, max_docs)


def eval_v0_window_texts(root, tok, n: int = 256, T: int = 512) -> list[str]:
    """The texts of Eval v0's fixed held-out windows (validation split, window seed 1234)."""
    d = TokenData("val", root)
    n = min(n, (len(d.tokens) - 1) // T)
    eot = d.meta.get("eot_id", 0)
    out = []
    for s in d.eval_windows(n, T):
        ids = [int(t) for t in d.tokens[s:s + T] if int(t) != eot]
        out.append(tok.decode(ids))
    return out


def report(train: dict, heldout_root: Path, *, lambada: bool = True, eval_v1: Path | None = None,
           jaccard: float = 0.8, ngram: int = 13, leak_threshold: float = 0.5, plant: int = 20,
           max_docs: int | None = None, n_lambada: int | None = None, splits: dict | None = None) -> dict:
    """``train`` maps a source name to its prepared folder; ``splits`` (optional) maps a source name to the split
    the mixture actually trains on (default "train"): a mixture that trains on a held-out split must be checked
    on that split, not on the folder's training split."""
    t0 = time.time()
    tok = load_tokenizer()
    out: dict = {"definitions": {"exact": "sha1 of NFC-normalized, whitespace-collapsed text",
                                 "near": f"MinHash 128 perms, 16 bands x 8 rows, word 5-gram shingles, exact Jaccard >= {jaccard}",
                                 "leak": f"word {ngram}-gram overlap fraction >= {leak_threshold}"}}
    # 1. provenance
    tok_sha = provenance.read_meta(DEFAULT_OUT).get("tokenizer_sha256")
    splits = splits or {}
    out["provenance"] = {name: [str(f) for f in provenance.check_source(root, splits.get(name, "train"), tokenizer_sha256=tok_sha)]
                         for name, root in train.items()}
    # texts
    tr = {name: texts_of(root, splits.get(name, "train"), tok, max_docs) for name, root in train.items()}
    held = {s: texts_of(heldout_root, s, tok) for s in ("val", "test")}
    out["sizes"] = {**{f"train:{k}": len(v) for k, v in tr.items()}, **{f"heldout:{k}": len(v) for k, v in held.items()}}
    # 2. exact
    held_hash = {s: {doc_hash(t) for t in v} for s, v in held.items()}
    out["exact_cross_split"] = {f"{name}->{s}": sum(doc_hash(t) in held_hash[s] for t in texts)
                                for name, texts in tr.items() for s in held}
    out["seconds_exact"] = round(time.time() - t0, 1)
    # 3. near
    t1 = time.time()
    groups = {f"train:{k}": v for k, v in tr.items()}
    groups.update({f"heldout:{k}": v for k, v in held.items()})
    nd = neardup.cross_split_near_dups(groups, threshold=jaccard)
    pairs = [p for p in nd["pairs"] if p["a"].startswith("train:") != p["b"].startswith("train:")]
    for p in pairs:                                      # keep a short excerpt for inspection
        a, b = groups[p["a"]][p["a_index"]], groups[p["b"]][p["b_index"]]
        p["excerpt_a"], p["excerpt_b"] = a[:160], b[:160]
    out["near_cross_split"] = {"candidates": nd["candidates"], "pairs_train_heldout": pairs,
                               "heldout_docs_with_train_near_dup": {
                                   s: len({p["b_index"] if p["b"] == f"heldout:{s}" else p["a_index"]
                                           for p in pairs if f"heldout:{s}" in (p["a"], p["b"])}) for s in held}}
    out["seconds_near"] = round(time.time() - t1, 1)
    # 4. leakage
    t2 = time.time()
    all_train = [t for v in tr.values() for t in v]
    index = leakage.NgramIndex(ngram).add_texts(all_train).finish()
    out["ngram_index_bytes"] = index.nbytes()
    lk = {"eval_v0_windows": leakage.leakage_report(index, eval_v0_window_texts(heldout_root, tok), leak_threshold, "eval_v0_windows")}
    lamb = []
    if lambada:
        from frontierlab.evals import suite_v0
        lamb = suite_v0.load_lambada(suite_v0.download_lambada(), n_lambada)
        lk["lambada"] = leakage.leakage_report(index, lamb, leak_threshold, "lambada")
    if eval_v1 is not None and (Path(eval_v1) / "val.bin").exists():
        lk["eval_v1_haystack_docs"] = leakage.leakage_report(index, texts_of(eval_v1, "val", tok), leak_threshold, "eval_v1_val")
    for r in lk.values():
        r.pop("overlaps")
    out["leakage"] = lk
    if lambada and plant:
        planted_items = lamb[:plant]
        pidx = leakage.NgramIndex(ngram).add_texts(leakage.plant(all_train[:5000], planted_items)).finish()
        rep = leakage.leakage_report(pidx, planted_items, leak_threshold, "planted")
        out["planted_control"] = {"planted": plant, "flagged": rep["flagged"], "recall": rep["flagged"] / plant}
    out["seconds_leakage"] = round(time.time() - t2, 1)
    out["seconds_total"] = round(time.time() - t0, 1)
    return out


def summary(rep: dict) -> str:
    lines = ["provenance:"]
    for name, fs in rep["provenance"].items():
        blocks = [f for f in fs if f.startswith("BLOCK")]
        warns = [f for f in fs if f.startswith("WARN")]
        lines.append(f"  {name}: {len(blocks)} block, {len(warns)} warn" + "".join(f"\n    {f}" for f in blocks + warns))
    lines.append(f"sizes: {rep['sizes']}")
    lines.append(f"exact cross-split duplicates: {rep['exact_cross_split']}")
    nc = rep["near_cross_split"]
    lines.append(f"near-duplicates train<->held-out: {len(nc['pairs_train_heldout'])} pairs "
                 f"({nc['candidates']} LSH candidates); held-out docs affected: {nc['heldout_docs_with_train_near_dup']}")
    for p in nc["pairs_train_heldout"][:5]:
        lines.append(f"    J={p['jaccard']:.3f} {p['a']}[{p['a_index']}] ~ {p['b']}[{p['b_index']}]: {p['excerpt_b'][:90]!r}")
    for k, r in rep["leakage"].items():
        lines.append(f"leakage {k}: {r['flagged']}/{r['items']} flagged (rate {r['flagged_rate']:.4f}), "
                     f"any 13-gram {r['any_ngram_rate']:.4f}, mean overlap {r['mean_overlap']:.5f}, too short {r['too_short']}")
    if "planted_control" in rep:
        lines.append(f"planted control: {rep['planted_control']}")
    lines.append(f"seconds: exact {rep['seconds_exact']}, near {rep['seconds_near']}, leakage {rep['seconds_leakage']}, total {rep['seconds_total']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", default=[f"edu={DEFAULT_OUT}"],
                    help="name=prepared_folder[:split] (the split the mixture trains on; default train)")
    ap.add_argument("--heldout", type=Path, default=DEFAULT_OUT, help="folder whose val/test are the held-out sets")
    ap.add_argument("--no-lambada", action="store_true")
    ap.add_argument("--eval-v1", type=Path, default=None)
    ap.add_argument("--jaccard", type=float, default=0.8)
    ap.add_argument("--ngram", type=int, default=13)
    ap.add_argument("--leak-threshold", type=float, default=0.5)
    ap.add_argument("--plant", type=int, default=20)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    train = dict(s.split("=", 1) for s in a.train)
    splits = {k: v.rsplit(":", 1)[1] for k, v in train.items() if ":" in v[2:]}      # name=root:split (not C:\)
    train = {k: (v.rsplit(":", 1)[0] if k in splits else v) for k, v in train.items()}
    rep = report(train, a.heldout, lambada=not a.no_lambada, eval_v1=a.eval_v1, jaccard=a.jaccard, ngram=a.ngram,
                 leak_threshold=a.leak_threshold, plant=a.plant, max_docs=a.max_docs, splits=splits)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rep, indent=2))
    print(summary(rep))


if __name__ == "__main__":
    main()
