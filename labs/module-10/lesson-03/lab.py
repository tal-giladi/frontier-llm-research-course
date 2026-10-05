"""Lab 10.3 — synthetic and rephrased data. Fill in the TODOs; run `pytest labs/module-10/lesson-03` to check."""

from __future__ import annotations

import re
from collections import Counter

NUM = re.compile(r"\d+(?:[.,]\d+)*")       # numbers; strip "," before comparing ("40,000" == "40000")
WORD = re.compile(r"[a-zA-Z]+")             # words, compared lowercase


def generation_flops(n_nonemb: int, layers: int, d_attn: int, prompt: int, generated: int,
                     head_flops_per_token: int = 0) -> float:
    """Forward FLOPs to prefill ``prompt`` tokens and generate ``generated`` tokens with a KV cache.

    2·N·(P + G)  +  2·L·d_attn·P²  +  4·L·d_attn·Σ_{c=P}^{P+G−1} c  +  head_flops_per_token·(P + G).
    Check: N=10, L=1, d_attn=2, P=3, G=2 gives 192.
    """
    raise NotImplementedError("TODO 1: generator compute")


def number_check(src: str, out: str) -> tuple[float | None, int]:
    """(share of the source's distinct numbers that appear in the output (None if the source has none),
    count of distinct numbers in the output that are not in the source)."""
    raise NotImplementedError("TODO 2: numbers kept and invented")


def novel_ngram_share(src: str, out: str, n: int = 4) -> float | None:
    """Share of the output's word n-grams (with repeats) that do not occur in the source; None if the output
    has fewer than n words. 0 means the output copies the source."""
    raise NotImplementedError("TODO 3: novelty")


def distinct_n(outputs: list[str], n: int = 2) -> float:
    """Distinct word n-grams over all outputs divided by total n-grams (0.0 if there are none)."""
    raise NotImplementedError("TODO 4: corpus diversity")


def cost_ratio(gen_flops: float, train_flops: float) -> float:
    """Generator FLOPs per training FLOP spent on the run that uses the data."""
    raise NotImplementedError("TODO 5: count the generator")


def decide(ci: tuple[float, float], margin: float = 0.0) -> str:
    """For (arm − baseline) loss: "adopt" if upper bound < −margin, "reject" if lower bound > −margin, else
    "inconclusive"."""
    raise NotImplementedError("TODO 6: the pre-stated rule")
