"""pass@k curves for the capability debate (lesson 14.4): does RL raise what the model *can* solve, or only
how often it solves what it already could?

Yue et al. (2025, Eq. 2) estimate pass@k per problem with the unbiased estimator 1 - C(n-c, k)/C(n, k) from
n samples, n equal to the largest k (up to 1,024 on AIME24/AMC23, 128 on MATH500 and GSM8K), and report that
RLVR models win at small k while base models reach a higher pass@k at large k. :func:`curve` computes the
mean curve, :func:`paired_curve_diff` the per-k difference between two models on the same problems with a
bootstrap interval over problems, and :func:`crossover` the first k at which the base model is at least as
good. ``coverage`` (pass@n) and the number of problems solved by one model but never by the other are the
direct form of the question.
"""

from __future__ import annotations

import numpy as np
import torch

from frontierlab.evals.suite_v2.core import pass_at_k


def curve(counts, n: int, ks) -> np.ndarray:
    """Mean pass@k over problems for each k; ``counts`` = correct samples per problem out of ``n``."""
    c = np.asarray(counts, dtype=int)
    return np.array([np.mean([pass_at_k(n, int(ci), k) for ci in c]) for k in ks])


def per_problem(counts, n: int, k: int) -> np.ndarray:
    return np.array([pass_at_k(n, int(ci), k) for ci in np.asarray(counts, dtype=int)])


def paired_curve_diff(counts_new, counts_base, n: int, ks, n_boot: int = 2000, seed: int = 0) -> list[dict]:
    """Per k: mean of (new - base) over problems and a 95% bootstrap interval (problems resampled)."""
    rng = np.random.default_rng(seed)
    P = len(counts_new)
    idx = rng.integers(0, P, size=(n_boot, P))
    out = []
    for k in ks:
        d = per_problem(counts_new, n, k) - per_problem(counts_base, n, k)
        boots = d[idx].mean(1)
        out.append({"k": int(k), "diff": float(d.mean()), "ci": (float(np.quantile(boots, 0.025)),
                                                                 float(np.quantile(boots, 0.975)))})
    return out


def crossover(curve_rl, curve_base, ks):
    """The smallest k with base >= RL, after RL was ahead at k = ks[0]; None if the curves never cross."""
    rl, base = np.asarray(curve_rl), np.asarray(curve_base)
    if rl[0] <= base[0]:
        return None
    for k, a, b in zip(ks, rl, base):
        if b >= a:
            return int(k)
    return None


def solved_sets(counts_a, counts_b) -> dict:
    a, b = np.asarray(counts_a) > 0, np.asarray(counts_b) > 0
    return {"both": int((a & b).sum()), "only_a": int((a & ~b).sum()), "only_b": int((~a & b).sum()),
            "neither": int((~a & ~b).sum())}


@torch.no_grad()
def toy_counts(policy, problems, n: int, max_new: int = 8, temperature: float = 1.0, seed: int = 4321,
               chunk: int = 4096) -> np.ndarray:
    """Correct samples out of ``n`` per toy problem (strict verifier), sampled in chunks with a fixed seed."""
    from frontierlab.posttrain.policy import sample
    from frontierlab.posttrain.tasks import encode_prompts, score
    g = torch.Generator().manual_seed(seed)
    rep = [p for p in problems for _ in range(n)]
    out = []
    for i in range(0, len(rep), chunk):
        part = rep[i:i + chunk]
        ro = sample(policy, encode_prompts(part), max_new, temperature, g)
        out.append(score(ro.response, part))
    return torch.cat(out).view(len(problems), n).sum(1).long().numpy()
