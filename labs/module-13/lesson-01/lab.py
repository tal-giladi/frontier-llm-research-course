"""Lab 13.1 — Open recipes as case studies. Fill in the TODOs; run `pytest labs/module-13/lesson-01` to check.

Part A builds the preference stage every open recipe uses (DPO and its two common variants) and the
Tülu 3 rule that turns ratings into pairs. Part B is the evidence audit: how large a stage's reported gain
must be before a benchmark of n items can distinguish it from item noise.
"""

from __future__ import annotations

import math

import numpy as np
import torch


# --------------------------------------------------------------------------- Part A: the preference stage

def dpo_loss(pol_c: torch.Tensor, pol_r: torch.Tensor, ref_c: torch.Tensor, ref_r: torch.Tensor, beta: float,
             len_c: torch.Tensor | None = None, len_r: torch.Tensor | None = None, normalise: bool = False,
             nll_coef: float = 0.0) -> torch.Tensor:
    """Mean DPO loss over B pairs from sequence log-probabilities (B,) of the chosen (c) and rejected (r)
    responses under the policy (pol_*) and the frozen reference (ref_*).

    h = beta * [(pol_c - ref_c) - (pol_r - ref_r)];  loss = mean(-log sigmoid(h))       (Rafailov et al. Eq. 7)
    normalise=True: divide each log-ratio by its response length first          (Tülu 3 Eq. 6)
    nll_coef > 0: add nll_coef * mean(-pol_c / len_c) (length-normalised NLL of the chosen; Llama 3 4.1.4)

    Use torch.nn.functional.logsigmoid, not log(sigmoid(.)), so large |h| does not underflow.
    """
    raise NotImplementedError("TODO 1: the DPO loss and its two variants")


def binarise(scores: list[float], rng: np.random.Generator) -> tuple[int, int] | None:
    """Tülu 3's binarisation of ratings (section 5.2.1): the chosen response is a highest-rated one (pick at
    random among ties) and the rejected one is drawn at random from those rated strictly lower. Return
    (chosen index, rejected index), or None when every response has the same rating (no pair).

    Draw the chosen index first, then the rejected one, each with ``rng.choice`` over the candidate indices
    in increasing order (the test replays the same random stream)."""
    raise NotImplementedError("TODO 2: ratings to a preference pair")


# --------------------------------------------------------------------------- Part B: the evidence audit

def binomial_se(p: float, n: int) -> float:
    """Standard error of an accuracy p measured on n independent items."""
    raise NotImplementedError("TODO 3: binomial standard error")


def mde_unpaired(p: float, n: int, z: float = 1.96) -> float:
    """Smallest difference between two independent accuracies near p, each on n items, that a two-sided
    test at the z level would call significant: z * sqrt(2 p (1 - p) / n)."""
    raise NotImplementedError("TODO 4: minimum detectable difference")


def audit(stages: list[tuple[str, float]], n_items: int | None, seed_spread: float | None = None) -> list[dict]:
    """For consecutive stages [(name, score in %), ...] return one row per transition:
    {"from", "to", "delta" (points), "mde" (points, from mde_unpaired at the *earlier* stage's accuracy and
    n_items, or None when n_items is None), "verdict"}.

    verdict: "beyond item noise" if mde is known and |delta| > mde; "within item noise" if mde is known and
    |delta| <= mde; otherwise (no item count, e.g. an average over benchmarks) compare with seed_spread:
    "beyond seed spread" if |delta| > seed_spread, "within seed spread" if not, and "no noise estimate" if
    seed_spread is None as well.
    """
    raise NotImplementedError("TODO 5: the stage-delta audit")
