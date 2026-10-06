"""A subset of IFEval's verifiable instructions (Zhou et al. 2023, arXiv 2311.07911) for the main path.

Pinned data: Hugging Face dataset ``google/IFEval`` at commit ``966cd89545d6b6acfd7638bc708b98261ca58e84``,
file ``ifeval_input_data.jsonl`` (207,111 bytes, SHA-256 below, Apache-2.0; checked 2026-10-06). The
paper describes 541 prompts and 25 instruction types.

This module implements 12 of the 25 types, the ones whose check is a few unambiguous lines of code.
A prompt is scored only if *all* its instructions are implemented; the result says how many prompts
were scored and skipped. For the full benchmark use lm-evaluation-harness's ``ifeval`` task (lm-eval
0.4.13 in the course pins) and check that the two agree on the prompts both score (a pilot check).

Metrics, as in the paper: prompt-level (every instruction of the prompt followed) and instruction-level
(fraction of instructions followed), each **strict** (the response as is) and **loose** (followed by
any of 8 variants of the response: identity, markdown ``*`` removed, first line removed, last line
removed, and their combinations).
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

IFEVAL_REPO = "google/IFEval"
IFEVAL_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
IFEVAL_URL = f"https://huggingface.co/datasets/{IFEVAL_REPO}/resolve/{IFEVAL_REVISION}/ifeval_input_data.jsonl"
IFEVAL_SHA256 = "6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2"


def _words(t):
    return re.findall(r"\b\w+\b", t)


def _cmp(n, relation, target):
    return n >= target if relation == "at least" else n < target


def no_comma(r, **kw):
    return "," not in r


def number_words(r, num_words, relation, **kw):
    return _cmp(len(_words(r)), relation, num_words)


def keyword_existence(r, keywords, **kw):
    return all(re.search(re.escape(k), r, flags=re.I) for k in keywords)


def forbidden_words(r, forbidden_words, **kw):
    return not any(re.search(r"\b" + re.escape(w) + r"\b", r, flags=re.I) for w in forbidden_words)


def keyword_frequency(r, keyword, frequency, relation, **kw):
    return _cmp(len(re.findall(re.escape(keyword), r, flags=re.I)), relation, frequency)


def english_lowercase(r, **kw):
    return r == r.lower()


def english_capital(r, **kw):
    return r == r.upper()


def end_checker(r, end_phrase, **kw):
    return r.strip().strip("\"").lower().endswith(end_phrase.strip().lower())


def quotation(r, **kw):
    s = r.strip()
    return len(s) > 1 and s[0] == "\"" and s[-1] == "\""


def json_format(r, **kw):
    s = r.strip()
    for fence in ("```json", "```Json", "```JSON", "```"):
        s = s.removeprefix(fence)
    s = s.removesuffix("```").strip()
    try:
        json.loads(s)
        return True
    except ValueError:
        return False


def postscript(r, postscript_marker, **kw):
    marker = postscript_marker.strip()
    pat = r"\s*p\.\s?p\.?\s?s.*$" if marker.lower() == "p.p.s" else r"\s*" + re.escape(marker.lower()) + r".*$"
    return bool(re.search(pat, r.lower(), flags=re.M))


def title(r, **kw):
    return any(t.strip() for t in re.findall(r"<<[^\n]+>>", r))


CHECKERS = {
    "punctuation:no_comma": no_comma,
    "length_constraints:number_words": number_words,
    "keywords:existence": keyword_existence,
    "keywords:forbidden_words": forbidden_words,
    "keywords:frequency": keyword_frequency,
    "change_case:english_lowercase": english_lowercase,
    "change_case:english_capital": english_capital,
    "startend:end_checker": end_checker,
    "startend:quotation": quotation,
    "detectable_format:json_format": json_format,
    "detectable_content:postscript": postscript,
    "detectable_format:title": title,
}


def loose_variants(r: str) -> list[str]:
    lines = r.split("\n")
    first_removed = "\n".join(lines[1:]).strip()
    last_removed = "\n".join(lines[:-1]).strip()
    both = "\n".join(lines[1:-1]).strip()
    base = [r, first_removed, last_removed, both]
    return base + [v.replace("*", "") for v in base]


def check(instruction_id: str, response: str, kwargs: dict, loose: bool = False) -> bool:
    fn = CHECKERS[instruction_id]
    kwargs = {k: v for k, v in (kwargs or {}).items() if v is not None}
    variants = loose_variants(response) if loose else [response]
    return any(v.strip() and fn(v, **kwargs) for v in variants)


def supported(item: dict) -> bool:
    return all(i in CHECKERS for i in item["instruction_id_list"])


def score_item(item: dict, response: str) -> dict:
    res = {}
    for mode in ("strict", "loose"):
        flags = [check(i, response, kw, loose=mode == "loose")
                 for i, kw in zip(item["instruction_id_list"], item["kwargs"])]
        res[f"prompt_{mode}"] = float(all(flags))
        res[f"inst_{mode}"] = flags
    return res


def download(dest: str | Path, timeout: float = 60.0) -> Path:
    dest = Path(dest)
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == IFEVAL_SHA256:
        return dest
    with urllib.request.urlopen(IFEVAL_URL, timeout=timeout) as r:   # noqa: S310 - pinned https URL
        data = r.read()
    if hashlib.sha256(data).hexdigest() != IFEVAL_SHA256:
        raise ValueError("IFEval file does not match the pinned SHA-256")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def load(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def components(items: list[dict], responses: list[str]) -> dict:
    """Eval v2 components for the main path, in item order (supported prompts only)."""
    rows = [score_item(it, r) for it, r in zip(items, responses) if supported(it)]
    inst_s = [float(f) for r in rows for f in r["inst_strict"]]
    inst_l = [float(f) for r in rows for f in r["inst_loose"]]
    return {"ifeval_prompt_strict": {"kind": "instruction", "items": [r["prompt_strict"] for r in rows]},
            "ifeval_prompt_loose": {"kind": "instruction", "items": [r["prompt_loose"] for r in rows]},
            "ifeval_inst_strict": {"kind": "instruction", "items": inst_s},
            "ifeval_inst_loose": {"kind": "instruction", "items": inst_l},
            "_ifeval_counts": {"scored": len(rows), "skipped": len(items) - len(rows)}}
