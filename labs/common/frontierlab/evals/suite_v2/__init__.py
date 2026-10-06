"""Eval Suite v2 (Module 12, lesson 12.4): run after every post-training stage, next to the task reward.

Three questions, each scored per item so two checkpoints can be compared with a paired bootstrap:

1. **Task** — did the trained skill improve? pass@1 from n samples and the unbiased pass@k
   (Chen et al. 2021, Eq. 1), plus greedy accuracy, on held-out prompts.
2. **Capability retention** — did anything else get worse? Skills the stage did not train (the toy:
   subtraction; the main path: held-out log-likelihood and a reference benchmark) and the
   teacher-forced loss on the previous stage's held-out data.
3. **Instruction following** — does the model still obey verifiable format instructions? IFEval-style
   checks (Zhou et al. 2023): prompt-level and instruction-level, strict (and loose on the main path).

:mod:`.core` holds the shared pieces (pass@k, paired comparisons, the retention guard, the report);
:mod:`.toy` is the free-CPU suite for the arithmetic world; :mod:`.ifeval` implements a subset of
IFEval's checkers for the main path, with a pinned copy of the data.
"""

from frontierlab.evals.suite_v2.core import VERSION, compare, pass_at_k, report

__all__ = ["VERSION", "compare", "pass_at_k", "report"]
