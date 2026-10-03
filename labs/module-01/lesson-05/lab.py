"""Lab 01.5 — the experiment record. Fill in the TODOs; run `pytest labs/module-01/lesson-05`.

You write the core of a run-card diff: flatten a nested card, decide which differences invalidate a
comparison, and compare tensors bit for bit. The full tool, with more rules, is
``python -m frontierlab.record`` (lesson 01.5); do not import it here.
"""

from __future__ import annotations

import torch

# Keys that never affect a result (bookkeeping). A key is covered if it equals one of these or starts
# with one of these followed by a dot.
BOOKKEEPING = ("run", "question", "parent_run", "args.run", "args.question", "args.parent", "args.log_every",
               "args.ckpt_every", "args.eval_every", "args.stop_after", "args.max_minutes", "args.device")
DATA_AND_EVAL = ("data", "data_files", "args.eval_windows", "args.seq")
SOFTWARE = ("git.commit", "hardware.torch", "hardware.python", "hardware.cuda", "hardware.gpu")


def flatten(card: dict, prefix: str = "") -> dict:
    """{"a": {"b": 1}} -> {"a.b": 1}. Keep empty dicts as values (``{"extra": {}}`` -> {"extra": {}})."""
    raise NotImplementedError("TODO 1: flatten a nested run card into dotted keys")


def classify(key: str, changed: tuple[str, ...] = (), seeds_are_replicates: bool = False,
             axis: str = "tokens") -> str:
    """What a difference in ``key`` means. Return one of: "ignore", "changed", "replicate", "warn", "invalidates".

    Apply in this order ("covered by" = equal to, or below, one of the listed keys):
      1. covered by BOOKKEEPING                              -> "ignore"
      2. covered by ``changed`` (the declared variable)      -> "changed"
      3. "args.seed" or "args.data_seed"                     -> "replicate" if seeds_are_replicates else "invalidates"
      4. covered by DATA_AND_EVAL                            -> "invalidates"
      5. in SOFTWARE                                         -> "invalidates" if axis == "wallclock" else "warn"
      6. "params..." keys                                    -> "changed" if any declared change is "config" or below it,
                                                                else "invalidates"
      7. "budget.tokens" or "args.steps"                     -> "invalidates" if axis == "tokens" else "changed"
      8. any other "config..." or "args..." key              -> "invalidates"
      9. anything else                                       -> "warn"
    """
    raise NotImplementedError("TODO 2: classify one difference")


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True only if shape, dtype and every bit agree.

    ``torch.equal`` is VALUE equality: it says 0.0 == -0.0 (different bits) and NaN != NaN (same bits).
    Compare the raw bytes instead: ``x.contiguous().reshape(-1).view(torch.uint8)``.
    """
    raise NotImplementedError("TODO 3: bitwise tensor equality")
