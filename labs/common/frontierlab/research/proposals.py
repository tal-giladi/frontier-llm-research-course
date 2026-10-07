"""Choosing what to work on (lesson 19.1).

A proposal is scored by the expected value of running it per unit of cost:

    score = value × p_decisive / cost_gpu_hours

* ``value`` (1–10): how much the answer would change a decision you or your team will make. A result that
  changes nothing is worth 0 however interesting it is.
* ``p_decisive`` (0–1]: the probability that the planned experiment gives an answer the decision rule can act
  on — the effect appears at the planned scale *and* the design can detect it (:func:`power`). A null result
  from a well-powered experiment is decisive; an underpowered run that shows nothing is not.
* ``cost_gpu_hours``: main-path GPU-hours including seeds, tuning and evaluation, labelled PROJECTED until
  measured.

The estimates are guesses, so a ranking is only as good as its robustness: :func:`rank_robustness` perturbs every
estimate by up to a stated factor and reports how often the top proposal stays on top. :func:`proxy_trend` reads a
cheap proxy (the same comparison at 2–3 small sizes, plan section 12.1) the way a pilot does: does the effect grow,
shrink, stay flat, flip sign or stay inside the noise along the ladder?

``CAPSTONE_CLAIMS`` is the Module 20 claim list. The costs are the course labs' own PROJECTED main-path figures
for one arm set and one seed (lesson named in ``cost_source``); a capstone needs seeds and an extension on top.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

CAPSTONE_CLAIMS = {
    "gspo-vs-grpo": {
        "title": "GSPO vs GRPO stability",
        "claim": "Sequence-level importance ratios and clipping (GSPO) train more stably than token-level GRPO.",
        "source": "Zheng et al., Group Sequence Policy Optimization, sections 4-5",
        "url": "https://arxiv.org/abs/2507.18071",
        "course": "14.1, 14.2",
        "lab_cost_gpu_h": (31.0, 52.0), "cost_source": "lesson 14.2 main path, PROJECTED",
        "proxy": "the 14.2 objective comparison on the 0.6B T4 variant, two seeds",
        "proxy_risk": "instability may need longer rollouts, larger models or MoE routing to appear",
    },
    "qkclip-vs-qknorm": {
        "title": "QK-Clip vs QK-norm",
        "claim": "QK-Clip bounds attention-logit growth under Muon as well as QK-norm does, without a loss cost.",
        "source": "Kimi K2 section 2.1 (QK-Clip); DeepSeek-V4 sections 2.3.3 and 2.4 (RMSNorm on q and KV)",
        "url": "https://arxiv.org/abs/2507.20534",
        "course": "07.2, 07.5",
        "lab_cost_gpu_h": (0.8, 0.8), "cost_source": "lesson 07.2 main path (7 arms, one seed, pilot-30m), PROJECTED",
        "proxy": "the 07.2 arms at toy size and raised learning rate",
        "proxy_risk": "Kimi's logit growth appeared at 9B activated parameters; tau = 100 was never reached at toy size",
    },
    "dsa-vs-dense": {
        "title": "DSA quality and cost vs dense",
        "claim": "Learned top-k sparse attention keeps dense quality while cutting long-context cost.",
        "source": "DeepSeek-V3.2 sections 2.1-2.2",
        "url": "https://arxiv.org/abs/2512.02556",
        "course": "05.2, 05.3",
        "lab_cost_gpu_h": (1.0, 1.0), "cost_source": "lesson 05.3 main path (profiling and Eval v1), PROJECTED",
        "proxy": "component profiling at 4K-32K and quality on Eval v1 at 1K-16K",
        "proxy_risk": "the O(L^2) indexer and unfused kernels can hide the crossover below 128K",
    },
    "mhc-vs-hc": {
        "title": "mHC vs HC stability",
        "claim": "Doubly-stochastic residual mixing (mHC) removes the instability of unconstrained hyper-connections.",
        "source": "Xie et al., mHC: Manifold-Constrained Hyper-Connections, sections 3-5",
        "url": "https://arxiv.org/abs/2512.24880",
        "course": "06.2",
        "lab_cost_gpu_h": (2.0, 3.5), "cost_source": "lesson 06.2 main path, pilot-30m rung, PROJECTED",
        "proxy": "HC vs mHC at 10M/30M with standard and raised learning rate",
        "proxy_risk": "the stability difference was reported at 3B-27B and may not appear at small scale",
    },
    "micro-anneal": {
        "title": "Micro-anneal data scoring",
        "claim": "Short anneals on a candidate data source rank sources the way full runs would.",
        "source": "OLMo 2, section 4.4.2 (microanneals)",
        "url": "https://arxiv.org/abs/2501.00656",
        "course": "10.4",
        "lab_cost_gpu_h": (1.3, 1.3), "cost_source": "lesson 10.4 main path, PROJECTED",
        "proxy": "micro-anneals from a toy checkpoint against full toy runs",
        "proxy_risk": "rankings from short anneals may not transfer to longer runs or larger models",
    },
    "onpolicy-distill": {
        "title": "On-policy vs off-policy distillation at equal compute",
        "claim": "On-policy distillation reaches a target score with less compute than SFT on teacher outputs.",
        "source": "Thinking Machines Lab, On-Policy Distillation (2025-10-27); Agarwal et al. (GKD)",
        "url": "https://thinkingmachines.ai/blog/on-policy-distillation/",
        "course": "13.2",
        "lab_cost_gpu_h": (5.0, 7.0), "cost_source": "lesson 13.2 main path, PROJECTED",
        "proxy": "the 13.2 toy distillation with teacher FLOPs counted",
        "proxy_risk": "the reported 9-30x saving is for a 32B teacher and an 8B student on AIME'24",
    },
}


@dataclass
class Proposal:
    claim_id: str                 # a key of CAPSTONE_CLAIMS, or "own:<slug>" for your own question
    question: str                 # one answerable sentence
    decision: str                 # what you will do differently depending on the answer
    value: float                  # 1-10
    p_decisive: float             # (0, 1]
    cost_gpu_h: float             # main-path GPU-hours, all seeds and tuning included
    proxy: str                    # the cheap first experiment
    kill: str                     # the result that would make you stop

    def to_dict(self) -> dict:
        return asdict(self)


def score(p: Proposal) -> float:
    """value × p_decisive / cost (the expected decision value per GPU-hour)."""
    return p.value * p.p_decisive / p.cost_gpu_h


def rank(props: list[Proposal]) -> list[tuple[Proposal, float]]:
    """Proposals with their scores, best first (ties keep the input order)."""
    scored = [(p, score(p)) for p in props]
    return sorted(scored, key=lambda t: -t[1])


def rank_robustness(props: list[Proposal], factor: float = 2.0, n: int = 4000, seed: int = 0) -> float:
    """Fraction of random perturbations under which the top-ranked proposal stays top.

    Every value, p_decisive and cost is multiplied independently by ``exp(u)``, ``u ~ Uniform(-ln f, ln f)``
    (log-uniform within a factor ``f``); p_decisive is capped at 1. A fraction near 1 means the decision does not
    hinge on the guesses; near 1/len(props) means it is a coin toss and a cheap proxy is worth more than more
    thinking.
    """
    if len(props) < 2:
        return 1.0
    top = max(range(len(props)), key=lambda i: score(props[i]))
    rng = np.random.default_rng(seed)
    v = np.array([p.value for p in props], dtype=np.float64)
    q = np.array([p.p_decisive for p in props], dtype=np.float64)
    c = np.array([p.cost_gpu_h for p in props], dtype=np.float64)
    lf = math.log(factor)
    m = np.exp(rng.uniform(-lf, lf, size=(n, 3, len(props))))
    s = v * m[:, 0] * np.minimum(q * m[:, 1], 1.0) / (c * m[:, 2])
    return float((s.argmax(axis=1) == top).mean())


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def power(effect: float, seed_std: float, n_per_arm: int, z_alpha: float = 1.96) -> float:
    """Probability that a two-arm comparison with ``n_per_arm`` seeds detects a true difference ``effect``
    (two-sided, alpha 0.05, normal approximation; the inverse of ``stats.min_detectable_effect``)."""
    if seed_std <= 0:
        return 1.0 if effect != 0 else 0.0
    se = seed_std * math.sqrt(2.0 / n_per_arm)
    z = abs(effect) / se
    return _phi(z - z_alpha) + _phi(-z - z_alpha)


def proxy_trend(sizes, effects, noise: float, k: float = 2.0) -> str:
    """Read a cheap proxy along a scale ladder (plan section 12.1).

    ``effects`` are the measured differences at ``sizes`` (same sign convention throughout), ``noise`` the seed std
    of one difference. Returns, in this order of precedence:

    * ``"below noise"`` — every |effect| < k·noise: the proxy says nothing either way;
    * ``"sign flips"`` — effects above the noise point in different directions: the small scale is not a proxy;
    * ``"grows"`` / ``"shrinks"`` — |effect| changes by more than k·noise from the smallest to the largest size;
    * ``"flat"`` — otherwise.
    """
    order = np.argsort(np.asarray(sizes, dtype=np.float64))
    e = np.asarray(effects, dtype=np.float64)[order]
    if e.size == 0:
        raise ValueError("need at least one size")
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


def check_proposals(props: list[Proposal], require_capstone: bool = True) -> list[str]:
    """Problems with a set of ranked proposals (empty list: acceptable)."""
    probs = []
    if len(props) != 3:
        probs.append(f"need exactly 3 proposals, got {len(props)}")
    ids = [p.claim_id for p in props]
    if len(set(ids)) != len(ids):
        probs.append("proposals must address different claims")
    if require_capstone and not any(i in CAPSTONE_CLAIMS for i in ids):
        probs.append("at least one proposal must use a claim from the capstone list")
    for p in props:
        tag = p.claim_id
        if p.claim_id not in CAPSTONE_CLAIMS and not p.claim_id.startswith("own:"):
            probs.append(f"{tag}: unknown claim id (use a CAPSTONE_CLAIMS key or 'own:<slug>')")
        if not 1 <= p.value <= 10:
            probs.append(f"{tag}: value must be 1-10")
        if not 0 < p.p_decisive <= 1:
            probs.append(f"{tag}: p_decisive must be in (0, 1]")
        if p.cost_gpu_h <= 0:
            probs.append(f"{tag}: cost must be positive")
        for field in ("question", "decision", "proxy", "kill"):
            if not str(getattr(p, field)).strip():
                probs.append(f"{tag}: {field} is empty")
        if p.question.strip() and not p.question.strip().endswith("?"):
            probs.append(f"{tag}: the question should be one answerable question ending in '?'")
    return probs
