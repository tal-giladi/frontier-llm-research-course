"""Lab 18.3 — frontier evaluation: time horizons, contamination and the v3 verdict. Fill in the TODOs; run
`pytest labs/module-18/lesson-03` to check. ``eval_lab.py`` uses these functions on METR's released runs and on
your toy model.
"""

from __future__ import annotations

import math


def horizon_from_fit(a: float, b: float, p: float = 0.5) -> float:
    """The task length (human minutes) at which a fitted curve p(success) = sigmoid(a + b * log2(minutes)) equals
    ``p``. Solve a + b * log2(t) = log(p / (1 - p)) for t. Return math.inf when b >= 0 (success does not fall
    with task length, so the curve never crosses ``p`` from above)."""
    raise NotImplementedError("TODO 1: invert the logistic fit")


def palm_contaminated(item: str, train_ngrams: set, n: int = 8, threshold: float = 0.7) -> bool:
    """PaLM's rule (as cited in the Llama 2 report): an item is contaminated if at least ``threshold`` of its
    character n-grams (``item[i:i+n]`` as tuples of characters) occur in ``train_ngrams``. An item shorter than
    ``n`` has no n-grams and is not contaminated."""
    raise NotImplementedError("TODO 2: the n-gram overlap rule")


def min_k_percent(token_logprobs: list[float], k: float = 0.2) -> float:
    """Min-K% Prob (Shi et al. 2023): the mean of the lowest ceil(k * len) token log-probabilities."""
    raise NotImplementedError("TODO 3: the membership score")


def v3_verdict(passes_v2: bool, base_contamination_passes: bool, new_contamination_passes: bool,
               benchmark_flags: list[str]) -> str:
    """The Eval v3 decision for one comparison, in this order:
    "do not report: retired benchmark" if any flag starts with "retired";
    "fail: contamination" if either side fails its contamination check;
    "fail: regression" if the v2 rule failed;
    "pass with flags" if there are other flags; otherwise "pass"."""
    raise NotImplementedError("TODO 4: the v3 verdict")
