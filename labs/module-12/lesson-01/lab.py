"""Lab 12.1 — rewards. Fill in the TODOs; run `pytest labs/module-12/lesson-01` to check."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import gammaln


def bt_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry negative log-likelihood, averaged over pairs: -log sigmoid(r_c - r_r).

    Use a numerically stable form (``F.logsigmoid``), not ``torch.log(torch.sigmoid(...))``.
    """
    raise NotImplementedError("TODO 1: Bradley-Terry loss")


def bon_kl(n: int) -> float:
    """KL(best-of-n || base) in nats: log n - (n - 1) / n."""
    raise NotImplementedError("TODO 2: best-of-n KL")


def bon_expected(proxy, value, n: int) -> float:
    """Unbiased E[value of the response that ``proxy`` ranks first among n draws], from a pool of N.

    Sort the pool by ``proxy`` (ascending). The i-th smallest (1-based) is the pick of a uniformly random
    n-subset with probability C(i-1, n-1) / C(N, n). Compute the weights in log space with ``gammaln``
    (log C(a, b) = gammaln(a+1) - gammaln(b+1) - gammaln(a-b+1)); they are 0 for i < n.
    """
    raise NotImplementedError("TODO 3: unbiased best-of-n estimator")


def ece(p, y, bins: int = 10) -> float:
    """Expected calibration error: 10 equal-width bins of predicted probability p on [0, 1] (the last
    bin includes 1.0); sum over non-empty bins of (bin share of items) * |mean p - mean y|."""
    raise NotImplementedError("TODO 4: calibration error")


def verifier_errors(accepted, truly_correct) -> dict:
    """Error rates of a verifier against the truth.

    ``false_positive_rate`` = accepted among truly wrong / truly wrong; ``false_negative_rate`` = rejected
    among truly correct / truly correct; ``precision`` = truly correct among accepted / accepted. Use
    ``max(1, denominator)`` so empty groups give 0.
    """
    raise NotImplementedError("TODO 5: verifier false positives and negatives")


def peak(n_values, gold_values, tol: float = 0.0) -> dict:
    """Where the gold reward of best-of-n peaks: {"n_peak", "gold_peak", "gold_last", "declined"} where
    ``declined`` is True if the last value is below the peak by more than ``tol``."""
    raise NotImplementedError("TODO 6: the over-optimisation point")
