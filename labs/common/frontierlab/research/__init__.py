"""Research practice (Module 19): choosing, reproducing and writing up experiments.

Plain Python and NumPy, so every rule can be read in a few minutes. Four modules:

* :mod:`~frontierlab.research.proposals` (lesson 19.1) — the capstone claim list, proposals scored by
  value × probability of a decisive answer ÷ cost, how robust a ranking is to wrong estimates, the power of a
  planned comparison, and the trend of a cheap proxy along a scale ladder.
* :mod:`~frontierlab.research.reproduce` (lesson 19.2) — Wortsman et al.'s learning-rate sensitivity, the
  reproduction decision against a tolerance stated before the runs (direction and magnitude separately), the
  deviation log and the reproduction record with its validator.
* :mod:`~frontierlab.research.writeup` (lesson 19.3) — a claims register parsed from a write-up, a linter that
  checks each claim against the run cards it cites (seeds, budget parity, checkpoint selection, scope,
  uncertainty), and a linter for figure specifications (one claim, uncertainty, budget-matched axes).
* :mod:`~frontierlab.research.contract` (Module 19 project) — the experiment contract as data, its validator
  and the rubric total.
"""

from frontierlab.research import contract, proposals, reproduce, writeup

__all__ = ["contract", "proposals", "reproduce", "writeup"]
