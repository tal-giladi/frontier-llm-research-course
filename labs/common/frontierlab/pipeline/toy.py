"""The toy arithmetic world (Module 12's :mod:`frontierlab.posttrain.tasks`) as data for pipeline stages.

* :func:`example` turns a problem and a response text into an :class:`~frontierlab.pipeline.seqs.Example`
  (BOS + prompt, response + EOS).
* :func:`sample_texts` draws n responses per problem from a policy (Module 12's KV-cache sampler) and
  charges the ledger for them.
* :func:`aspect_score` is a Tülu-3-style multi-aspect rating made of two programmatic aspects (is the
  number right, is the format the tag asked for), averaged: 0, 0.5 or 1.
* :func:`build_pairs` makes preference pairs the way Tülu 3 binarises ratings (section 5.2.1, stage 3):
  the highest-rated response is chosen and the rejected one is drawn at random from the responses rated
  lower. Prompts whose responses all have the same rating give no pair.
* :func:`eval_v2` runs Eval Suite v2 (lesson 12.4) on a policy.

Prompts for every stage come from the *training* triples (``split_problems``); Eval v2 uses the held-out
triples, so no stage ever trains on an evaluation prompt.
"""

from __future__ import annotations

import numpy as np
import torch

from frontierlab.pipeline.compute import Ledger, n_params
from frontierlab.pipeline.seqs import Example
from frontierlab.posttrain.policy import sample
from frontierlab.posttrain.tasks import (TAGS, Problem, encode_prompts, follows_instruction, make_problems,
                                         response_text, split_problems)
from frontierlab.posttrain.tokenizer import BOS, EOS, TOK


def example(problem: Problem, text: str, finished: bool = True) -> Example:
    return Example.of([BOS] + TOK.encode(problem.prompt), TOK.encode(text) + ([EOS] if finished else []))


def gold_examples(problems: list) -> list:
    return [example(p, p.target) for p in problems]


def train_problems(n: int, seed: int, ops: str = "+-", tags: str = TAGS, digits: int = 2) -> list:
    """``n`` problems from the training triples only."""
    _, held = split_problems(digits)
    return make_problems(n, digits, ops=ops, tags=tags, seed=seed, exclude=held)


@torch.no_grad()
def sample_texts(model, problems: list, n: int, temperature: float = 1.0, seed: int = 0, max_new: int = 8,
                 ledger: Ledger | None = None, category: str = "student_sample", chunk: int = 4096) -> list:
    """For each problem, ``n`` sampled (text, finished) pairs. Problems must share one digit count."""
    gen = torch.Generator().manual_seed(seed)
    rep = [p for p in problems for _ in range(n)]
    out = []
    for i in range(0, len(rep), chunk):
        part = rep[i:i + chunk]
        prompts = encode_prompts(part)
        ro = sample(model, prompts, max_new, temperature, gen)
        if ledger is not None:
            ledger.forward(category, n_params(model), float(prompts.numel() + ro.mask.sum()))
        out += [response_text(r) for r in ro.response]
    return [out[j * n:(j + 1) * n] for j in range(len(problems))]


def number_right(problem: Problem, text: str) -> bool:
    digits = "".join(ch for ch in text if ch.isdigit())
    return bool(digits) and int(digits) == problem.result


def aspect_score(problem: Problem, text: str, finished: bool) -> float:
    """Mean of two ratings: the number is right (any format), and the response obeys the tag's format."""
    return 0.5 * float(finished and number_right(problem, text)) + 0.5 * float(follows_instruction(text, finished, problem))


def build_pairs(problems: list, candidates: list, scorer, seed: int = 0, extra: list | None = None) -> tuple[list, list]:
    """Preference pairs [(chosen Example, rejected Example)] and their metadata.

    ``candidates[i]`` is a list of (text, finished) for ``problems[i]``; ``extra[i]`` (optional) adds more
    candidates (for example a refusal written by a template rather than sampled). ``scorer(problem, text,
    finished) -> float``. Duplicate texts are scored once."""
    rng = np.random.default_rng(seed)
    pairs, meta = [], []
    for i, (p, cands) in enumerate(zip(problems, candidates)):
        pool = dict()
        for text, fin in list(cands) + list(extra[i] if extra else []):
            pool.setdefault((text, fin), None)
        keys = list(pool)
        scores = np.array([scorer(p, t, f) for t, f in keys], dtype=float)
        if scores.max() == scores.min():
            continue
        top = np.flatnonzero(scores == scores.max())
        low = np.flatnonzero(scores < scores.max())
        c, r = keys[int(rng.choice(top))], keys[int(rng.choice(low))]
        pairs.append((example(p, c[0], c[1]), example(p, r[0], r[1])))
        meta.append({"problem": p, "chosen": c, "rejected": r, "s_chosen": float(scores.max()),
                     "s_rejected": float(scores[keys.index(r)])})
    return pairs, meta


def shuffle_labels(pairs: list, seed: int = 0, p: float = 0.5) -> list:
    """The random-preference control: each pair's chosen and rejected are swapped with probability p."""
    rng = np.random.default_rng(seed)
    return [(r, c) if rng.random() < p else (c, r) for c, r in pairs]


def eval_v2(model, n_items: int = 200) -> dict:
    """Eval Suite v2 for the toy world (lesson 12.4), on held-out triples."""
    from frontierlab.evals.suite_v2.toy import run_suite
    return run_suite(model, n_items=n_items)


GUARDS = {"sub_greedy": 0.02, "sft_nll": 0.02, "if_strict": 0.02, "if_correct": 0.02}
