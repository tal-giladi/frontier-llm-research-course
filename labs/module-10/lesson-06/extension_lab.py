"""Lab 10.6 (extension): per-language pipelines and repository / task-family splits. Analysis only, ~10 min CPU.

    python -m frontierlab.datax.sources prepare fw2-fra --docs 2000    # FineWeb2 French, 5.8 MB, ~30 s
    python -m frontierlab.datax.sources prepare fw2-deu --docs 2000    # German
    python -m frontierlab.datax.sources prepare fw2-heb --docs 2000    # Hebrew
    python labs/module-10/lesson-06/extension_lab.py

Part A, multilingual: (1) fertility of Data-v0's English tokenizer and of Qwen3's multilingual tokenizer on
English, French, German and Hebrew documents; (2) stop-word filters: Gopher's English rule (at least 2 of 8 English stop
words) on every language, a stop-word list per language from word frequencies, and your Quantile threshold;
(3) the MinHash cluster sizes FineWeb2 stores with every document, and rehydration weights computed from the
removal rates of that filter.

Part B, code and tasks: (4) Python source files of installed packages (pinned versions in the venv), split
by file, by directory and by package, with the share of held-out files that have a near-copy (MinHash,
Jaccard >= 0.7) in train; package licences from their metadata; (5) one shard of SWE-smith (pinned
revision, 4.1 MB): instance-level vs repository-level split, and how many held-out bug patches have a
near-copy in train.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from importlib import metadata
from pathlib import Path

import numpy as np

from frontierlab.data.loader import TokenData
from frontierlab.data.prepare import DEFAULT_OUT
from frontierlab.datax import groups
from frontierlab.datax.neardup import decode_docs
from frontierlab.datax.sources import M10_ROOT, load_tokenizer
from frontierlab.labkit import load_path

HERE = Path(__file__).resolve().parent
QWEN, QWEN_REV = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"
SWE, SWE_REV, SWE_FILE = "SWE-bench/SWE-smith", "ea6d7173829c7ec8fa16c22055699ff2e9188091", "data/train-00002-of-00011.parquet"
PACKAGES = ["transformers", "numpy", "jinja2", "httpx", "httpcore", "click", "anyio", "aiohttp", "attrs", "fsspec",
            "datasets", "huggingface_hub", "yaml", "idna", "packaging", "tokenizers"]
LANGS = {"eng (edu)": DEFAULT_OUT, "fra": M10_ROOT / "fw2-fra", "deu": M10_ROOT / "fw2-deu", "heb": M10_ROOT / "fw2-heb"}


def part_a(lab, tok, n_docs: int):
    from transformers import AutoTokenizer
    qtok = AutoTokenizer.from_pretrained(QWEN, revision=QWEN_REV)
    texts = {k: decode_docs(TokenData("train", r), tok, n_docs) for k, r in LANGS.items() if (Path(r) / "train.bin").exists()}
    print("(1) fertility: tokens per byte / per word")
    for k, t in texts.items():
        d0 = groups.fertility(lambda s: tok.encode(s).ids, t)
        q = groups.fertility(lambda s: qtok.encode(s, add_special_tokens=False), t)
        print(f"  {k:10s} bytes/char {d0['bytes_per_char']:.2f}  Data-v0 tok: {lab.tokens_per_byte(lambda s: tok.encode(s).ids, t):.3f}/byte "
              f"{d0['tokens_per_word']:.2f}/word   Qwen3 tok: {lab.tokens_per_byte(lambda s: qtok.encode(s, add_special_tokens=False), t):.3f}/byte "
              f"{q['tokens_per_word']:.2f}/word")
    print("(2) stop-word filters: the English list on every language, then a list per language")
    gopher = {"the", "be", "to", "of", "and", "that", "have", "with"}
    words = {k: [[w.lower() for w in groups._WORD.findall(x)] for x in t] for k, t in texts.items()}
    own = {}
    for k, ws in words.items():
        c = Counter(w for doc in ws for w in doc if w.isalpha())
        tot = sum(c.values())
        own[k] = {w for w, n in c.items() if n / tot >= 0.008}          # frequency threshold, as FineWeb2 4.4.1
    ratio = {k: np.array([sum(w in own[k] for w in d) / max(1, len(d)) for d in ws]) for k, ws in words.items()}
    eng_thr = float(np.quantile(ratio["eng (edu)"], 0.05))               # an English threshold removing 5%
    removed = {}
    for k, ws in words.items():
        rm_eng = np.mean([len(gopher & set(d)) < 2 for d in ws])
        rm_own = np.mean([len(own[k] & set(d)) < 2 for d in ws])
        thr = lab.quantile_threshold(ratio["eng (edu)"], eng_thr, ratio[k])
        removed[k] = ratio[k] < thr
        print(f"  {k:10s} own list {len(own[k]):2d} words ({', '.join(sorted(own[k])[:6])}...); '>= 2 Gopher English stop words' "
              f"removes {rm_eng:.3f}; '>= 2 of its own' removes {rm_own:.3f}; stop-word ratio: English threshold {eng_thr:.3f} "
              f"removes {groups.removal_rate(ratio[k], lo=eng_thr):.3f}, Quantile threshold {thr:.3f} removes {removed[k].mean():.3f}")
    print("(3) MinHash cluster sizes recorded by FineWeb2, and rehydration weights from the Quantile filter's removal rates")
    for k in ("fra", "deu", "heb"):
        root = LANGS[k]
        if k not in removed or not (Path(root) / "train_provenance.jsonl").exists():
            continue
        rows = [json.loads(x) for x in (Path(root) / "train_provenance.jsonl").read_text(encoding="utf-8").splitlines()]
        cs = np.minimum(np.array([int(r.get("minhash_cluster_size", 1)) for r in rows]), 6)[:removed[k].size]
        w = groups.rehydration_weights(cs, removed[k])
        best = min(w["removal_rate"].values())
        print(f"  {k}: documents per cluster size {dict(sorted(Counter(cs.tolist()).items()))} (6 = 6+); removal rate "
              f"{ {s: round(r, 3) for s, r in w['removal_rate'].items()} }, global {w['global_removal_rate']:.3f}; weights "
              f"{ {s: round(lab.rehydration_weight(r, w['global_removal_rate'], best), 2) for s, r in w['removal_rate'].items()} }")


def py_files(max_per_pkg: int):
    import site
    sp = Path(next(p for p in site.getsitepackages() if p.endswith("site-packages")))
    rows = []
    for pkg in PACKAGES:
        base = sp / pkg
        if not base.exists():
            continue
        files = sorted(base.rglob("*.py"))
        if pkg == "transformers":
            files = sorted((base / "models").rglob("*.py"))
        for f in files[:max_per_pkg]:
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if len(text) < 400:
                continue
            rel = f.relative_to(sp)
            rows.append({"pkg": pkg, "dir": str(rel.parent), "file": str(rel), "text": text})
    return rows


def licence_of(pkg: str) -> str:
    dist = {"yaml": "PyYAML", "attrs": "attrs", "huggingface_hub": "huggingface_hub", "jinja2": "Jinja2"}.get(pkg, pkg)
    try:
        md = metadata.metadata(dist)
    except metadata.PackageNotFoundError:
        return "?"
    lic = md.get("License-Expression") or md.get("License") or ""
    if not lic or len(lic) > 40:
        cl = [c.split("::")[-1].strip() for c in md.get_all("Classifier") or [] if c.startswith("License ::")]
        lic = ", ".join(cl) or (lic[:40] + "..." if lic else "?")
    return f"{lic} ({md.get('Version')})"


def part_b(lab, max_per_pkg: int):
    rows = py_files(max_per_pkg)
    pk = Counter(r["pkg"] for r in rows)
    print(f"(4) {len(rows)} Python files from {len(pk)} packages: {dict(pk)}")
    print("    licences:", {p: licence_of(p) for p in pk})
    texts = [r["text"] for r in rows]
    for unit in ("file", "dir", "pkg"):
        sp = [lab.group_split(r[unit], 150, 150, salt="m10") for r in rows]
        res = groups.cross_split_rate(texts, sp, threshold=0.7)
        print(f"    split by {unit:4s}: held-out files {res['heldout_items']:4d}, with a near-copy in train "
              f"{res['heldout_with_train_near_copy']:4d} ({res['rate']:.3f})")
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    p = hf_hub_download(SWE, SWE_FILE, repo_type="dataset", revision=SWE_REV)
    t = pq.read_table(p, columns=["instance_id", "patch"]).to_pydict()
    keys = [lab.swesmith_keys(i) for i in t["instance_id"]]
    print(f"(5) SWE-smith {SWE_FILE} @ {SWE_REV[:10]}: {len(keys)} instances, {len({k for k, _ in keys})} repositories, "
          f"families {Counter(f for _, f in keys).most_common(5)}")
    rng = np.random.default_rng(0)
    take = np.sort(rng.choice(len(keys), size=min(1500, len(keys)), replace=False))
    patches = [t["patch"][i] for i in take]
    for unit, kk in (("instance", [t["instance_id"][i] for i in take]), ("repository", [keys[i][0] for i in take])):
        sp = [lab.group_split(k, 150, 150, salt="m10") for k in kk]
        res = groups.cross_split_rate(patches, sp, threshold=0.7)
        repos_both = len({keys[i][0] for i, s in zip(take, sp) if s != "train"} & {keys[i][0] for i, s in zip(take, sp) if s == "train"})
        print(f"    split by {unit:10s}: held-out {res['heldout_items']:4d}, near-copy patch in train "
              f"{res['heldout_with_train_near_copy']:4d} ({res['rate']:.3f}); repositories on both sides {repos_both}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", type=int, default=2000)
    ap.add_argument("--max-per-pkg", type=int, default=120)
    ap.add_argument("--part", choices=["a", "b", "both"], default="both")
    a = ap.parse_args(argv)
    import sys
    sys.stdout.reconfigure(encoding="utf-8")          # Hebrew and German words in the output on Windows consoles
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    t0 = time.time()
    if a.part in ("a", "both"):
        part_a(lab, load_tokenizer(), a.docs)
        print(f"part A {time.time() - t0:.0f}s")
    t1 = time.time()
    if a.part in ("b", "both"):
        part_b(lab, a.max_per_pkg)
        print(f"part B {time.time() - t1:.0f}s")


if __name__ == "__main__":
    main()
