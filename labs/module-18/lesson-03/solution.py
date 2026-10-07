"""Reference solution for lab 18.3 — frontier evaluation."""

from __future__ import annotations

import math


def horizon_from_fit(a: float, b: float, p: float = 0.5) -> float:
    if b >= 0:
        return math.inf
    return 2.0 ** ((math.log(p / (1 - p)) - a) / b)


def palm_contaminated(item: str, train_ngrams: set, n: int = 8, threshold: float = 0.7) -> bool:
    grams = [tuple(item[i:i + n]) for i in range(len(item) - n + 1)]
    if not grams:
        return False
    return sum(g in train_ngrams for g in grams) / len(grams) >= threshold


def min_k_percent(token_logprobs: list[float], k: float = 0.2) -> float:
    lp = sorted(token_logprobs)
    m = max(1, math.ceil(k * len(lp)))
    return sum(lp[:m]) / m


def v3_verdict(passes_v2: bool, base_contamination_passes: bool, new_contamination_passes: bool,
               benchmark_flags: list[str]) -> str:
    if any(f.startswith("retired") for f in benchmark_flags):
        return "do not report: retired benchmark"
    if not (base_contamination_passes and new_contamination_passes):
        return "fail: contamination"
    if not passes_v2:
        return "fail: regression"
    return "pass with flags" if benchmark_flags else "pass"
