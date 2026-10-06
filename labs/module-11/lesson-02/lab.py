"""Lab 11.2 — predicting downstream capability. Fill in the TODOs; run `pytest labs/module-11/lesson-02` to check.

Inputs are float64 NumPy arrays or torch tensors as stated. The references are in ``frontierlab.scaling.downstream``.
"""

from __future__ import annotations

import numpy as np
import torch


def choice_metrics(lp: torch.Tensor, answer: torch.Tensor, cont: int) -> dict:
    """``lp``: (n, k) float64 total log-probability of each option; ``answer``: (n,) int64 index of the correct one;
    ``cont``: tokens per option. Return a dict of floats (means over items):

    * ``acc``: fraction of items whose correct option has the largest log-probability;
    * ``p_correct``: mean softmax probability (over the k options) of the correct option;
    * ``nll_correct``: mean of −lp[correct] / cont (per-token NLL of the correct option).
    """
    raise NotImplementedError("TODO 1: multiple-choice metrics from option log-probabilities")


def sigmoid_curve(x, x0: float, s: float, lo: float, hi: float) -> np.ndarray:
    """lo + (hi − lo) / (1 + exp(s·(x − x0))): accuracy as a decreasing function of task NLL x (s > 0)."""
    raise NotImplementedError("TODO 2: the step-2 curve")


def exact_match_from_token_acc(p, k: int) -> np.ndarray:
    """If each of k tokens is right independently with probability p, the chance that all k are right."""
    raise NotImplementedError("TODO 3: an all-or-nothing metric from a per-token one")


def explained_variance(scores: np.ndarray) -> np.ndarray:
    """Fraction of variance explained by each principal component of a (models × benchmarks) matrix with no
    missing values: standardise every column (subtract the mean, divide by the population std, ddof=0), take the
    singular values s of the result (np.linalg.svd), return s² / Σ s², largest first."""
    raise NotImplementedError("TODO 4: PCA of benchmark scores")


def two_step(nll_target: float, sigmoid_params: dict) -> float:
    """Step 2 of the two-step prediction: the accuracy the fitted sigmoid gives at the predicted task NLL.
    ``sigmoid_params`` has keys x0, s, lo, hi (use your sigmoid_curve)."""
    raise NotImplementedError("TODO 5: from predicted task loss to predicted accuracy")
