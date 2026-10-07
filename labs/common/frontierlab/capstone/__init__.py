"""Module 20 — the capstone: reproduce a claim at small scale, extend it, defend it.

* :mod:`~frontierlab.capstone.claims` — the six claims learners choose from, each checked against its primary
  source, with the earlier lessons and code it builds on, budgets per variant (main path PROJECTED with the formula
  :func:`~frontierlab.capstone.claims.projected_gpu_hours`), what a sound null looks like and a suggested extension (20.1).
* :mod:`~frontierlab.capstone.uncertainty` — paired hierarchical bootstrap over seeds and items, the paired seed
  t-interval, the decision rule with an equivalence margin, the noise floor (20.1).
* :mod:`~frontierlab.capstone.package` — the capstone package format and :func:`check_package`: contract filled,
  run cards with traceable parents, budget and tuning parity, uncertainty reported, claims within evidence (20.1).
  ``python -m frontierlab.capstone.package <dir>`` exits 1 on any problem.
* :mod:`~frontierlab.capstone.scaffold` — one claim end to end on CPU (QK-Clip vs unclipped Muon, extended to QK-norm),
  through the unmodified course loop (20.1).
* :mod:`~frontierlab.capstone.review` — reviewer questions, the written defence check, the revision log check (20.2).
* :mod:`~frontierlab.capstone.sample` — a constructed sample capstone with planted weaknesses (20.2).

Nothing here judges whether a method won: every check is about soundness, and a sound null result passes.
"""

from frontierlab.capstone.claims import CLAIMS, Claim, get, projected_gpu_hours
from frontierlab.capstone.package import Problem, check_package
from frontierlab.capstone.uncertainty import decide, hierarchical_bootstrap, noise_floor, seed_t_interval

__all__ = ["CLAIMS", "Claim", "Problem", "check_package", "decide", "get", "hierarchical_bootstrap", "noise_floor",
           "projected_gpu_hours", "seed_t_interval"]
