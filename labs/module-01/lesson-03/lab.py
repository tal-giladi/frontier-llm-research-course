"""Lab 01.3 — designing an experiment. Fill in the TODOs; run `pytest labs/module-01/lesson-03`.

Three small functions that every later lab uses: matching budgets on a comparison axis, selecting
hyperparameters without touching the test split, and applying a decision rule written down BEFORE
the results.
"""

from __future__ import annotations

from frontierlab.flops import flops_per_token
from frontierlab.model.config import ModelConfig


def matched_steps(axis: str, ref: ModelConfig, other: ModelConfig, ref_steps: int, T: int,
                  ref_tok_per_s: float | None = None, other_tok_per_s: float | None = None) -> int:
    """Training steps for ``other`` so that it matches ``ref`` (trained ``ref_steps``) on ``axis``.

    Both arms use the same batch size and sequence length T, so steps are proportional to tokens.

    * ``"tokens"``    -> the same number of steps.
    * ``"flops"``     -> the same training FLOPs: scale by flops_per_token(ref) / flops_per_token(other).
    * ``"wallclock"`` -> the same seconds: scale by other_tok_per_s / ref_tok_per_s (measured, same hardware).
    * ``"params"``    -> raise ValueError: equal parameters constrains the MODELS, not the training
      budget; you must still choose one of the other three axes for the run length.

    Round to the nearest integer.
    """
    raise NotImplementedError("TODO 1: match the budget on the chosen axis")


def select_then_report(val_scores: dict, test_scores: dict) -> tuple[str, float]:
    """Pick the configuration with the LOWEST validation loss and return (name, its test loss).

    ``val_scores`` and ``test_scores`` map configuration name -> loss. The test split is read exactly
    once, for the chosen configuration; choosing on the test split is the leak this function prevents.
    """
    raise NotImplementedError("TODO 2: select on validation, report on test")


def decide(ci: tuple[float, float], min_gain: float) -> str:
    """Apply a pre-stated rule to the 95% CI of (new - baseline) loss. Lower loss is better.

    * "adopt"        if the whole interval is below -min_gain (better by at least min_gain);
    * "reject"       if the whole interval is above -min_gain (not better by min_gain, maybe worse);
    * "inconclusive" otherwise (the interval contains -min_gain): more seeds or a bigger effect needed.
    """
    raise NotImplementedError("TODO 3: the decision rule")
