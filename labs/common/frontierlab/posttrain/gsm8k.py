"""GSM8K for the main path: pinned files, answer extraction and a strict verifier.

Pinned source (checked 2026-10-06): Hugging Face dataset ``openai/gsm8k`` at commit
``740312add88f781978c0658806c59bc2815b9866``, config ``main``: ``main/train-00000-of-00001.parquet``
(2,306,545 bytes) and ``main/test-00000-of-00001.parquet`` (419,088 bytes), MIT licence. The SHA-256 of
each file is the LFS object id on the Hub, checked after download.

Gold answers end with ``#### <number>``. The policy is prompted (few-shot, from the *training* split) to
end its solution the same way, and :func:`strict_reward` accepts only a response whose text up to the
first stop sequence contains ``#### <number>`` with the right value. :func:`last_number_reward` is the
lenient rule many scripts use (the last number anywhere in the text), kept to measure how often the two
disagree (lesson 12.1).
"""

from __future__ import annotations

import hashlib
import re
import urllib.request
from pathlib import Path

REPO, REVISION = "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866"
FILES = {"train": ("main/train-00000-of-00001.parquet", "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d"),
         "test": ("main/test-00000-of-00001.parquet", "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59")}
STOP = "\n\nQuestion:"
_NUM = r"-?[\d,]*\.?\d+"


def download(split: str, root: str | Path = "labs/common/data/m12/gsm8k", timeout: float = 60.0) -> Path:
    name, sha = FILES[split]
    dest = Path(root) / Path(name).name
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == sha:
        return dest
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}"
    with urllib.request.urlopen(url, timeout=timeout) as r:   # noqa: S310 - pinned https URL
        data = r.read()
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError(f"GSM8K {split} file does not match the pinned SHA-256")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def load(split: str, root: str | Path = "labs/common/data/m12/gsm8k") -> list[dict]:
    import pyarrow.parquet as pq
    t = pq.read_table(download(split, root)).to_pylist()
    return [{"question": r["question"], "solution": r["answer"], "answer": gold_answer(r["answer"])} for r in t]


def normalise(num: str) -> str | None:
    s = num.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else repr(v)


def gold_answer(solution: str) -> str:
    return normalise(solution.split("####")[-1])


def cut(response: str) -> str:
    """The response up to the first stop sequence (a base model keeps writing new questions)."""
    return response.split(STOP)[0]


def strict_answer(response: str) -> str | None:
    m = re.search(r"####\s*(" + _NUM + ")", cut(response))
    return normalise(m.group(1)) if m else None


def strict_reward(response: str, answer: str) -> float:
    return float(strict_answer(response) == answer)


def last_number_reward(response: str, answer: str) -> float:
    nums = re.findall(_NUM, cut(response))
    return float(bool(nums) and normalise(nums[-1]) == answer)


def few_shot_prompt(question: str, shots: list[dict]) -> str:
    """``Question: ...\\nAnswer: <solution with #### n>`` for each shot, then the question."""
    parts = [f"Question: {s['question']}\nAnswer: {s['solution']}" for s in shots]
    return "\n\n".join(parts + [f"Question: {question}\nAnswer:"])
