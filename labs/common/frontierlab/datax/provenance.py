"""Provenance and licence records for every training source and mixture (lesson 10.1).

    python -m frontierlab.datax.provenance check labs/common/data/v0 labs/common/data/m10/web
    python -m frontierlab.datax.provenance manifest runs/m10/mix.json --out runs/m10/manifest.json

A **source record** answers, for one prepared folder: where did every token come from (dataset,
config, pinned revision, the exact command), under which licence (and what it obliges: attribution,
share-alike), how big it is (documents and tokens per split), and whether the files on disk are the
ones that were recorded (SHA-256). A **manifest** is the list of source records of a mixture, with
the weights, the tokenizer they share and a licence summary; it goes next to the run card of every
run trained on that mixture.

:func:`check_source` returns findings at three levels:

* ``BLOCK`` — the source must not be used as is: unpinned revision (a branch name or nothing), no
  licence, an unknown licence, files whose hash differs from the record, a different tokenizer;
* ``WARN`` — usable, but someone must decide: share-alike licence, a source that is a held-out split,
  token counts that differ from the file size;
* ``OK`` — checks that passed, so the report shows what was checked.

The licence table below is information for engineers, not legal advice: it records what each licence
is commonly understood to require, and an unknown licence is a block until a person has looked.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from frontierlab.data.prepare import DEFAULT_OUT, sha256_file

LICENSES = {
    "ODC-By 1.0": {"attribution": True, "share_alike": False, "url": "https://opendatacommons.org/licenses/by/1-0/"},
    "CC BY-SA 3.0 and GFDL": {"attribution": True, "share_alike": True, "url": "https://creativecommons.org/licenses/by-sa/3.0/"},
    "CC BY-SA 4.0": {"attribution": True, "share_alike": True, "url": "https://creativecommons.org/licenses/by-sa/4.0/"},
    "CC BY 4.0": {"attribution": True, "share_alike": False, "url": "https://creativecommons.org/licenses/by/4.0/"},
    "MIT": {"attribution": True, "share_alike": False, "url": "https://opensource.org/license/mit"},
    "Apache-2.0": {"attribution": True, "share_alike": False, "url": "https://www.apache.org/licenses/LICENSE-2.0"},
    "BSD-3-Clause": {"attribution": True, "share_alike": False, "url": "https://opensource.org/license/bsd-3-clause"},
    "PSF-2.0": {"attribution": True, "share_alike": False, "url": "https://docs.python.org/3/license.html"},
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
DATA_V0_PROVENANCE = {"dataset": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT",
                      "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "license": "ODC-By 1.0",
                      "command": "python -m frontierlab.data.prepare"}


@dataclass
class Finding:
    level: str      # BLOCK | WARN | OK
    source: str
    message: str

    def __str__(self):
        return f"{self.level:5s} {self.source}: {self.message}"


def read_meta(root: str | Path) -> dict:
    return json.loads((Path(root) / "meta.json").read_text())


def source_record(root: str | Path, split: str = "train") -> dict:
    """The provenance record of one prepared folder (Data-v0 gets its known provenance filled in)."""
    root = Path(root)
    meta = read_meta(root)
    prov = dict(meta.get("provenance") or {})
    if not prov and meta.get("name") == "Data-v0":
        prov = dict(DATA_V0_PROVENANCE)
    for k in ("dataset", "config", "revision", "license"):
        prov.setdefault(k, meta.get(k))
    lic = LICENSES.get(prov.get("license") or "")
    return {"root": str(root), "name": meta.get("name"), "split": split, "dataset": prov.get("dataset"),
            "config": prov.get("config"), "revision": prov.get("revision"), "license": prov.get("license"),
            "license_url": prov.get("license_url") or (lic or {}).get("url"),
            "attribution_required": (lic or {}).get("attribution"), "share_alike": (lic or {}).get("share_alike"),
            "command": prov.get("command"), "parent": meta.get("parent"), "selection": meta.get("selection"),
            "tokenizer_sha256": meta.get("tokenizer_sha256"), "vocab_size": meta.get("vocab_size"),
            "files": {s: meta[s] for s in ("train", "val", "test") if s in meta}}


def check_source(root: str | Path, split: str = "train", verify_hashes: bool = True,
                 tokenizer_sha256: str | None = None) -> list[Finding]:
    root = Path(root)
    rec = source_record(root, split)
    name = rec["name"] or root.name
    out: list[Finding] = []
    rev = rec["revision"] or ""
    out.append(Finding("OK" if HEX40.match(rev) else "BLOCK", name,
                       f"revision {rev or 'missing'}" + ("" if HEX40.match(rev) else " is not a pinned 40-hex commit")))
    lic = rec["license"]
    if not lic:
        out.append(Finding("BLOCK", name, "no licence recorded"))
    elif lic not in LICENSES:
        out.append(Finding("BLOCK", name, f"licence {lic!r} is not in the reviewed table; a person must review it"))
    else:
        out.append(Finding("OK", name, f"licence {lic} (attribution {'required' if rec['attribution_required'] else 'not required'})"))
        if rec["share_alike"]:
            out.append(Finding("WARN", name, f"{lic} is share-alike: derived datasets that include this text carry the licence"))
    if split in ("val", "test"):
        out.append(Finding("WARN", name, f"the source is a held-out split ({split}): training on it leaks evaluation data"))
    if tokenizer_sha256 and rec["tokenizer_sha256"] != tokenizer_sha256:
        out.append(Finding("BLOCK", name, "tokenized with a different tokenizer than the mixture"))
    for s, info in rec["files"].items():
        path = root / f"{s}.bin"
        if not path.exists():
            if s == split:
                out.append(Finding("BLOCK", name, f"{path} missing"))
            continue
        n = path.stat().st_size // 2
        if "tokens" in info and info["tokens"] != n:
            out.append(Finding("WARN", name, f"{s}: meta says {info['tokens']} tokens, file has {n}"))
        if verify_hashes and "bin_sha256" in info:
            ok = sha256_file(path) == info["bin_sha256"]
            out.append(Finding("OK" if ok else "BLOCK", name, f"{s}.bin SHA-256 {'matches' if ok else 'DIFFERS from'} the record"))
    return out


def build_manifest(spec, verify_hashes: bool = True) -> dict:
    """Manifest of a :class:`frontierlab.datax.mixture.MixtureSpec`: one record per source, findings, licences."""
    tok = read_meta(spec.sources[0].root).get("tokenizer_sha256")
    total = sum(s.weight for s in spec.sources)
    sources, findings = [], []
    for s in spec.sources:
        rec = source_record(s.root, s.split)
        rec.update(mixture_name=s.name, weight=s.weight / total)
        sources.append(rec)
        findings += check_source(s.root, s.split, verify_hashes, tok)
    return {"mixture": spec.name, "mixture_digest": spec.digest(), "tokenizer_sha256": tok, "sources": sources,
            "attribution": sorted({f"{r['dataset']} ({r['license']})" for r in sources if r["attribution_required"]}),
            "share_alike_sources": [r["mixture_name"] for r in sources if r["share_alike"]],
            "blocked": [str(f) for f in findings if f.level == "BLOCK"],
            "warnings": [str(f) for f in findings if f.level == "WARN"],
            "checks": [str(f) for f in findings]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("roots", nargs="+", type=Path)
    c.add_argument("--no-hash", action="store_true")
    m = sub.add_parser("manifest")
    m.add_argument("mixture", type=Path)
    m.add_argument("--out", type=Path, default=None)
    m.add_argument("--no-hash", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        tok = read_meta(DEFAULT_OUT).get("tokenizer_sha256") if (DEFAULT_OUT / "meta.json").exists() else None
        bad = 0
        for r in a.roots:
            for f in check_source(r, verify_hashes=not a.no_hash, tokenizer_sha256=tok):
                print(f)
                bad += f.level == "BLOCK"
        raise SystemExit(1 if bad else 0)
    from frontierlab.datax.mixture import MixtureSpec
    man = build_manifest(MixtureSpec.load(a.mixture), verify_hashes=not a.no_hash)
    text = json.dumps(man, indent=2)
    if a.out:
        a.out.write_text(text)
    for line in man["checks"]:
        print(line)
    print(f"blocked: {len(man['blocked'])}  warnings: {len(man['warnings'])}  share-alike: {man['share_alike_sources']}")
    raise SystemExit(1 if man["blocked"] else 0)


if __name__ == "__main__":
    main()
