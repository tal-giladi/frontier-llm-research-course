"""Lab 19.1 — research taste and problem choice. Fill in the TODOs; run `pytest labs/module-19/lesson-01` to check.
``choose_lab.py`` uses these functions and your PROPOSALS.
"""

from __future__ import annotations

from frontierlab.research.proposals import power  # noqa: F401  (you will need it in TODO 2)

# TODO 4: your three proposals, best first after you have scored them. Each is a dict with the keys
#   claim_id    a key of frontierlab.research.proposals.CAPSTONE_CLAIMS, or "own:<slug>"; at least one from the list
#   question    one answerable question ending in "?"
#   decision    what you will do differently depending on the answer
#   value       1-10: how much the answer changes that decision
#   effect      the smallest difference that would change the decision, in the primary metric's units
#   seed_std    the seed noise of that difference's arms at the planned scale (measured, or labelled an assumption)
#   n_seeds     seeds per arm you can afford
#   p_transfer  0-1: probability the planned scale gives the answer that holds at the decision's scale
#   cost_gpu_h  main-path GPU-hours for every arm, seed, tuning run and evaluation (PROJECTED)
#   proxy       the cheap first experiment
#   kill        the result that would make you stop
PROPOSALS: list[dict] = []


def score(value: float, p_decisive: float, cost_gpu_h: float) -> float:
    """Expected decision value per GPU-hour: value x p_decisive / cost."""
    raise NotImplementedError("TODO 1: score a proposal")


def p_decisive(effect: float, seed_std: float, n_seeds: int, p_transfer: float) -> float:
    """Probability the experiment settles the decision: the power to detect ``effect`` with ``n_seeds`` per arm
    (``power`` from frontierlab.research.proposals) times the probability that the answer transfers."""
    raise NotImplementedError("TODO 2: probability of a decisive answer")


def proxy_trend(sizes, effects, noise: float, k: float = 2.0) -> str:
    """Read a cheap proxy along a scale ladder. Sort by size; then, in this order:
    "below noise" if every |effect| < k*noise; "sign flips" if the effects with |effect| >= k*noise have both signs;
    "grows" if |effect at the largest size| - |effect at the smallest| > k*noise; "shrinks" if < -k*noise;
    otherwise "flat"."""
    raise NotImplementedError("TODO 3: what does the ladder say?")
