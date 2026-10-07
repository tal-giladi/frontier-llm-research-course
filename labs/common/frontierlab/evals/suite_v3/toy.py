"""Eval Suite v3 for the toy arithmetic world (free CPU path).

    python -m frontierlab.evals.suite_v3.toy score runs/m13/project/toy-s0/s3/policy.pt --out runs/m18/eval/s3.json
    python -m frontierlab.evals.suite_v3.toy compare runs/m18/eval/a.json runs/m18/eval/b.json

v3 = every v2 component (unchanged items, pins and decision rule) plus a contamination section on its own item
sets, drawn from the held-out triples with a fixed seed:

* ``published``: 200 plain-addition items, the set a report would publish scores on;
* ``fresh``: 200 more items from the same held-out pool and distribution, never published (the GSM1k idea).

For every published item: ``overlap`` (char 6-gram overlap with every training text the caller declares, PaLM's
70% rule), ``min_k`` (Min-K% Prob of prompt + answer under the model, k = 20%) and ``correct`` (greedy). The section
reports the flagged count, the membership AUC of ``min_k`` between published and fresh items, and the fresh gap
(published minus fresh accuracy, with an interval). The rule: a model **fails the contamination check** if any
published item is flagged by overlap, or the fresh-gap interval lies above ``gap_guard`` (default 0.05).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from frontierlab.evals.suite_v2.toy import greedy_responses, run_suite as run_v2
from frontierlab.evals.suite_v3 import contamination as C
from frontierlab.evals.suite_v3.core import VERSION, attach, compare, report
from frontierlab.posttrain.sft import load_policy
from frontierlab.posttrain.tasks import Problem, encode_sft, problems_from, score, split_problems


def v3_items(digits: int = 2, n: int = 200, seed: int = 1818) -> tuple[list[Problem], list[Problem]]:
    """(published, fresh): disjoint random samples of the held-out plain-addition problems."""
    _, held = split_problems(digits)
    pool = problems_from(held, ".", digits, "+")
    random.Random(seed).shuffle(pool)
    return pool[:n], pool[n:2 * n]


def item_text(p: Problem) -> str:
    return p.prompt + p.target


@torch.no_grad()
def item_logprobs(model, problems: list[Problem]) -> list[list[float]]:
    """Per item, the log-probability of every token after BOS (prompt, answer and EOS)."""
    model.eval()
    ids, _ = encode_sft(problems)
    lp = torch.log_softmax(model(ids).logits[:, :-1].double(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    out = []
    for row, tgt in zip(lp, ids[:, 1:]):
        keep = tgt != 0
        out.append(row[keep].tolist())
    return out


def contamination_section(model, training_texts: list[str] | None, digits: int = 2, n: int = 200, k: float = 0.2,
                          gap_guard: float = 0.05, ngram: int = 6) -> dict:
    pub, fresh = v3_items(digits, n)
    if training_texts:
        index = C.build_index(training_texts, ngram)
        overlap = C.palm_rule([item_text(p) for p in pub], index, ngram)
    else:
        overlap = [False] * len(pub)
    mk_pub = [C.min_k_prob(x, k) for x in item_logprobs(model, pub)]
    mk_fresh = [C.min_k_prob(x, k) for x in item_logprobs(model, fresh)]
    corr_pub = score(greedy_responses(model, pub), pub).tolist()
    corr_fresh = score(greedy_responses(model, fresh), fresh).tolist()
    gap = C.fresh_gap(corr_pub, corr_fresh)
    flagged = sum(overlap)
    fails = flagged > 0 or gap["ci"][0] > gap_guard
    return {"declared_training_texts": len(training_texts or []), "ngram": ngram, "k": k, "gap_guard": gap_guard,
            "items": {"overlap": overlap, "min_k": mk_pub, "correct": corr_pub},
            "fresh": {"min_k": mk_fresh, "correct": corr_fresh},
            "flagged": flagged, "membership_auc": C.membership_auc(mk_pub, mk_fresh), "fresh_gap": gap,
            "passes": not fails}


def run_suite(model, training_texts: list[str] | None = None, benchmarks: tuple = (), **v2_kw) -> dict:
    res = run_v2(model, **v2_kw)
    return attach(res, contamination_section(model, training_texts), benchmarks)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("policy")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--training-texts", type=Path, help="a file with one declared training text per line")
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("new")
    c.add_argument("--guard", type=float, default=0.02)
    a = ap.parse_args(argv)
    if a.cmd == "score":
        texts = a.training_texts.read_text().splitlines() if a.training_texts else None
        res = run_suite(load_policy(a.policy), texts)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res))
        print(json.dumps(res["summary"], indent=2))
    else:
        base, new = json.loads(Path(a.base).read_text()), json.loads(Path(a.new).read_text())
        print(report(compare(base, new, default_guard=a.guard)))


if __name__ == "__main__":
    main()
