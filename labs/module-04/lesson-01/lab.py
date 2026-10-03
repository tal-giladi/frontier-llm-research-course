"""Lab 04.1 — the parts of Eval Suite v1 that decide whether it measures anything.

Fill in the TODOs; ``pytest labs/module-04/lesson-01`` checks them against the reference in
``frontierlab.evals.suite_v1``. ``check_eval.py`` then runs your functions on real items.
"""

from __future__ import annotations

import math

import torch


def oracle_answer(ids: list[int], remember: list[int], keys: set[int], candidates: list[int]) -> int | None:
    """Solve a synthetic item exactly from its token ids, or return None if the evidence is not there.

    The context contains statements ``remember + [a, b] + period`` (``remember`` is the token list of
    " Remember:"). ``ids[-1]`` is the query key. Follow the links a -> b starting from the query key
    (use the *first* statement for each key ``a`` that is in ``keys``) until you reach a token that is
    in ``candidates``; return it. Return None if the chain breaks (or loops) before reaching a candidate.
    Statements with another prefix (" Ignore:") must be ignored.
    """
    raise NotImplementedError("TODO 1: read the Remember statements and follow the chain from ids[-1]")


def candidate_score(last_logits: torch.Tensor, candidates: list[int], answer: int) -> tuple[bool, float]:
    """(correct, logp_cand) for one item from the logits at its last position, shape (V,).

    correct: the answer has strictly the highest logit among the candidates.
    logp_cand: log-probability of the answer after renormalising the softmax over the candidates only
    (chance is ln(1/K) for K candidates).
    """
    raise NotImplementedError("TODO 2: restrict the logits to the candidates and score the answer")


@torch.no_grad()
def context_gain(model, windows: torch.Tensor, W: int, lo: int, hi: int) -> torch.Tensor:
    """Per window: mean loss of targets at positions lo..hi-1 with context cut to W tokens, minus with full context.

    ``windows`` (n, T) long: the first T tokens of n documents. Target token t is predicted from
    tokens 0..t-1 (full) and from tokens lo-W..t-1 (cut; at least W tokens of context). Use
    ``model(x, labels=x).per_token_loss`` (n, len-1), whose column p is the loss of predicting token p+1.
    Returns (n,) float: positive = the tokens more than W back helped.
    """
    raise NotImplementedError("TODO 3: two forward passes, then line up the target columns")


def effective_length(cells: list[dict], threshold: float, key: str = "acc") -> int | None:
    """Longest swept length L such that the CI lower bound of ``key`` is >= threshold at L and every shorter length.

    ``cells``: [{"length": L, key: (mean, lo, hi)}, ...] in any order. None if the shortest length fails.
    """
    raise NotImplementedError("TODO 4: sort by length and stop at the first failure")


def items_needed(p_model: float, p_chance: float, z_alpha: float = 1.96, z_power: float = 0.84) -> int:
    """Items needed to tell accuracy ``p_model`` from chance ``p_chance`` with ~80% power (alpha 0.05, two-sided).

    Normal approximation: n = ((z_alpha * sqrt(p0 (1 - p0)) + z_power * sqrt(p1 (1 - p1))) / (p1 - p0))^2,
    rounded up, with p0 = chance and p1 = the model's accuracy.
    """
    raise NotImplementedError("TODO 5: the sample-size formula above")


_ = math  # math is available for TODO 5
