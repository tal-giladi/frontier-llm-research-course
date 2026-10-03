"""The experiment record (Module 1, lesson 01.5): comparing run cards and checkpoints.

* :func:`diff_cards` — every difference between two ``run_card.yaml`` files, classified as
  ``invalidates`` (the runs are not comparable), ``changed`` (the declared variable under test),
  ``replicate`` (the seed, when seeds are the replicate axis), ``warn`` (record it, may matter) or
  ``ignore`` (bookkeeping).
* :func:`state_diff` — tensors that differ between two checkpoints, bit for bit.

    python -m frontierlab.record runs/a runs/b --changed config.attention --axis tokens
"""

from frontierlab.record.diff import Finding, classify, comparable, diff_cards, flatten, state_diff

__all__ = ["Finding", "classify", "comparable", "diff_cards", "flatten", "state_diff"]
