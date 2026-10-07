"""Reference solution for lab 19.1."""

from __future__ import annotations

import numpy as np

from frontierlab.research.proposals import power

# The reference proposals. Noise floors are ASSUMPTIONS for illustration (0.02 nats for a 30M-parameter rung is a
# guess, not a measurement): replace them with your own Module 1 project noise floor and the noise your earlier labs
# measured. Costs are the course labs' PROJECTED figures (frontierlab.research.proposals.CAPSTONE_CLAIMS) times the
# seeds and arms each proposal needs.
PROPOSALS = [
    {"claim_id": "micro-anneal",
     "question": "Do 2%-of-budget micro-anneals rank three candidate data sources in the same order as full anneals at pilot-30m?",
     "decision": "whether Data-v2 source choices can be made with micro-anneals instead of full anneals",
     "value": 7, "effect": 0.05, "seed_std": 0.02, "n_seeds": 2, "p_transfer": 0.6, "cost_gpu_h": 6.0,
     "proxy": "micro-anneals from the toy checkpoint against full toy anneals (lesson 10.4)",
     "kill": "the full anneals do not separate the sources beyond the noise, so there is no ranking to predict"},
    {"claim_id": "qkclip-vs-qknorm",
     "question": "At pilot-30m with Muon at a raised learning rate, does QK-Clip (tau 15) match QK-norm on held-out loss within 0.02 nats while bounding the max logit?",
     "decision": "whether Recipe-R with Muon keeps QK-norm or can use MLA with QK-Clip",
     "value": 6, "effect": 0.03, "seed_std": 0.02, "n_seeds": 3, "p_transfer": 0.5, "cost_gpu_h": 4.0,
     "proxy": "the 07.2 arms at toy size and raised learning rate, 2 widths",
     "kill": "the clip never binds at pilot-30m (max logit stays below tau) on the first seed"},
    {"claim_id": "gspo-vs-grpo",
     "question": "On the 1.7B base model with 8 rollouts per prompt, does GSPO cut the rate of reward collapses relative to GRPO at equal rollouts?",
     "decision": "which objective the course's reasoning-RL recipe uses",
     "value": 8, "effect": 0.15, "seed_std": 0.10, "n_seeds": 3, "p_transfer": 0.4, "cost_gpu_h": 40.0,
     "proxy": "the 14.2 comparison on the 0.6B T4 variant, two seeds",
     "kill": "neither objective collapses in 2 seeds at the proxy scale"},
]


def score(value: float, p_decisive: float, cost_gpu_h: float) -> float:
    return value * p_decisive / cost_gpu_h


def p_decisive(effect: float, seed_std: float, n_seeds: int, p_transfer: float) -> float:
    return power(effect, seed_std, n_seeds) * p_transfer


def proxy_trend(sizes, effects, noise: float, k: float = 2.0) -> str:
    order = np.argsort(np.asarray(sizes, dtype=np.float64))
    e = np.asarray(effects, dtype=np.float64)[order]
    big = e[np.abs(e) >= k * noise]
    if big.size == 0:
        return "below noise"
    if (big > 0).any() and (big < 0).any():
        return "sign flips"
    change = abs(e[-1]) - abs(e[0])
    if change > k * noise:
        return "grows"
    if change < -k * noise:
        return "shrinks"
    return "flat"
