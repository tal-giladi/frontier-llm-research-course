"""Eval Suite v3 (Module 18, lesson 18.3): Eval v2 plus contamination checks, benchmark lifecycle flags and
METR-style time horizons.

v2 asked three questions of a post-trained checkpoint: did the task improve, did anything else regress, does it
still follow instructions. v3 asks two more, of the evaluation itself:

4. **Can this score be trusted?** Contamination evidence per item (:mod:`.contamination`): n-gram overlap with the
   declared training data, a membership signal (Min-K% Prob) and a fresh, matched item set; a result that fails is
   reported, not compared.
5. **Is this benchmark still informative?** Its place in the lifecycle (:mod:`.lifecycle`): retired, saturated,
   audited errors, public test set.

:mod:`.horizon` adds the long-horizon measurement (METR's 50% time horizon with a hierarchical bootstrap), used on
METR's released runs on CPU. :mod:`.toy` is the free-CPU suite for the arithmetic world; :mod:`.hf` the main-path
contamination section for Hugging Face models; :mod:`.core` holds the result format and the comparison rule.
"""

from frontierlab.evals.suite_v3.core import VERSION, attach, compare, report

__all__ = ["VERSION", "attach", "compare", "report"]
