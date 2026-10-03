"""Lab 04.3 — data and budget for extending context by continued training.

Fill in the TODOs; ``pytest labs/module-04/lesson-03`` checks them against
``frontierlab.longctx.data`` and ``frontierlab.flops``.
"""

from __future__ import annotations

import numpy as np


def same_doc_context(doc_starts: np.ndarray, window_starts: np.ndarray, T: int) -> np.ndarray:
    """(n_windows, T) same-document context of every token of every window.

    ``doc_starts`` (sorted int array): offset of the first token of each document in the token stream.
    ``window_starts``: offset of the first token of each window. Entry [w, p] = how many earlier tokens of
    window w belong to the same document as its token p: min(p, offset - start of that token's document).
    Hint: ``np.searchsorted(doc_starts, offsets, side="right") - 1`` is the document of each offset.
    """
    raise NotImplementedError("TODO 1: same-document context per token")


def within_doc_start(doc_starts: np.ndarray, lengths: np.ndarray, T: int, u: int) -> int:
    """Map an integer u in [0, total) to the start of a window of T tokens inside one document.

    Only documents with ``lengths >= T`` count; a document of length n has n - T + 1 valid starts.
    Enumerate all valid starts document by document, in file order (document j's starts are
    doc_starts[j], doc_starts[j] + 1, ..., doc_starts[j] + n_j - T) and return the u-th one (0-based).
    Drawing u uniformly gives every valid window the same probability.
    """
    raise NotImplementedError("TODO 2: cumulative counts of valid starts, then find u's document")


def stage_plan(cfg, stages: list[tuple[int, float]], peak_flops: float, mfu: float) -> list[dict]:
    """Projected cost of a staged extension: one dict per stage (seq, tokens, flops, gpu_hours) plus a total.

    ``stages``: [(sequence length, training tokens), ...]. FLOPs per token at length T come from
    ``frontierlab.flops.flops_per_token(cfg, T)`` (training, includes the attention term, which grows
    with T). gpu_hours = flops / (peak_flops * mfu) / 3600. Append a final dict
    {"seq": "total", "tokens": ..., "flops": ..., "gpu_hours": ...}.
    """
    raise NotImplementedError("TODO 3: FLOPs per stage at that stage's length")


def decide(short_regression_ci: tuple[float, float], gain_ci: tuple[float, float], max_regression: float,
           min_gain: float) -> str:
    """The pre-stated decision rule of the lab's contract.

    ``short_regression_ci``: 95% CI of (extended - base) held-out loss at the original length (positive = worse).
    ``gain_ci``: 95% CI of the context gain at the new length (positive = the far context helps).
    "adopt" if the regression's upper bound <= max_regression AND the gain's lower bound >= min_gain;
    "reject" if the regression's lower bound > max_regression OR the gain's upper bound < min_gain;
    otherwise "inconclusive".
    """
    raise NotImplementedError("TODO 4: apply the rule exactly as written")
