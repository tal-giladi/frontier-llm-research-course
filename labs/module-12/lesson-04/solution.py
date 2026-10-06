"""Lab 12.4 — Eval Suite v2. Reference solution."""

from __future__ import annotations

import math
import re


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def follows_tag(text: str, finished: bool, tag: str, digits: int = 2) -> bool:
    if not finished:
        return False
    pattern = {".": r"\d+", "P": r"\d{%d}" % (digits + 1), "Q": r"#\d+#", "E": r"\d+!"}[tag]
    return bool(re.fullmatch(pattern, text))


def verdict(kind: str, lo: float, hi: float, guard: float) -> str:
    if kind == "task":
        return "improved" if lo > 0 else ("worse" if hi < 0 else "no clear change")
    return "regressed" if lo < -guard else "held"


def prompt_and_instruction_level(flags: list[list[bool]]) -> tuple[float, float]:
    prompt = sum(all(f) for f in flags) / len(flags)
    inst = sum(sum(f) for f in flags) / sum(len(f) for f in flags)
    return prompt, inst


def loose_variants(response: str) -> list[str]:
    lines = response.split("\n")
    base = [response, "\n".join(lines[1:]).strip(), "\n".join(lines[:-1]).strip(), "\n".join(lines[1:-1]).strip()]
    return base + [v.replace("*", "") for v in base]
