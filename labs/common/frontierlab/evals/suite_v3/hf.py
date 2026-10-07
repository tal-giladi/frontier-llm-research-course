"""Eval Suite v3 on the main path (Hugging Face models). Not run in this build except ``--smoke``; part of the
Module 18 pilot.

    python -m frontierlab.evals.suite_v3.hf score --model runs/m13/main/rlvr-s0/policy --chat --out runs/m18/main/v3/rlvr-s0.json
    python -m frontierlab.evals.suite_v3.hf compare runs/m18/main/v3/sft-s0.json runs/m18/main/v3/rlvr-s0.json
    python -m frontierlab.evals.suite_v3.hf score --smoke --out runs/m18/hf-smoke/v3.json                      # CPU

v3 = the main-path Eval v2 suite (:func:`frontierlab.pipeline.hf_eval.suite`: GSM8K pass@1 and greedy, LAMBADA
log-probability, the IFEval subset) plus a contamination section on the GSM8K test items it scores:

* ``overlap`` — word 8-gram overlap of each test question with the post-training data the course pipeline used
  (the pinned Tülu 3 / OLMo 2 SFT shard, the OLMo 2 preference shard and the GSM8K training split), PaLM's 70% rule.
  This is the only check that can fail the section: Module 13's data choices are declared, so they can be checked.
* ``min_k`` — Min-K% Prob of each test question, and of a **number-perturbed** copy of it (every number replaced
  by another of the same length, so the text has the same form but was never published). The AUC of original vs
  perturbed is a membership *signal* for the base model's undisclosed pretraining data, reported and never used as
  a verdict (Qwen3's pretraining data is not public; lesson 18.3).

There is no fresh-gap component on the main path: a perturbed problem has no known answer. The documented
alternative for contamination-sensitive work is allenai/OLMo-2-0425-1B, whose pretraining data is public (plan
decision 9).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch

from frontierlab.evals.suite_v3 import contamination as C
from frontierlab.evals.suite_v3.core import attach, compare, report


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def perturb_numbers(text: str, rng: random.Random) -> str:
    """Replace every number by a different random number with the same number of digits."""
    def sub(m):
        s = m.group(0)
        lo = 10 ** (len(s) - 1) if len(s) > 1 else 0
        for _ in range(20):
            v = str(rng.randrange(lo, 10 ** len(s)))
            if v != s:
                return v
        return s
    return re.sub(r"\d+", sub, text)


@torch.no_grad()
def text_logprobs(model, tok, texts: list[str]) -> list[list[float]]:
    from frontierlab.pipeline.hf_eval import encode
    dev = next(model.parameters()).device
    out = []
    for t in texts:
        ids = torch.tensor([encode(tok, t)], device=dev)
        lp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1).gather(-1, ids[0, 1:, None]).squeeze(-1)
        out.append(lp.tolist())
    return out


def declared_texts(smoke: bool) -> list[str]:
    if smoke:
        return ["natalia sold clips to 48 of her friends in april and then she sold half as many clips in may"]
    from frontierlab.pipeline.hf_stages import _rows, single_turn
    from frontierlab.posttrain import gsm8k
    texts = []
    for r in _rows("sft"):
        st = single_turn(r["messages"])
        if st:
            texts.append(st[0] + " " + st[1])
    for r in _rows("pref"):
        texts.append(str(r.get("prompt") or r.get("chosen") or ""))
    texts += [q["question"] + " " + q["solution"] for q in gsm8k.load("train")]
    return texts


def contamination_section(model, tok, questions: list[str], smoke: bool = False, n: int = 8, k: float = 0.2) -> dict:
    index: set = set()
    for t in declared_texts(smoke):
        index |= C.ngrams(words(t), n)
    overlap = C.palm_rule([words(q) for q in questions], index, n)
    rng = random.Random(0)
    pert = [perturb_numbers(q, rng) for q in questions]
    mk = [C.min_k_prob(x, k) for x in text_logprobs(model, tok, questions)]
    mk_p = [C.min_k_prob(x, k) for x in text_logprobs(model, tok, pert)]
    return {"declared": "Module 13 SFT and preference shards, GSM8K train", "ngram": n, "k": k,
            "items": {"overlap": overlap, "min_k": mk, "correct": []}, "fresh": {"min_k": mk_p, "correct": []},
            "flagged": int(sum(overlap)), "membership_auc": C.membership_auc(mk, mk_p),
            "fresh_gap": {"gap": float("nan"), "ci": (float("nan"), float("nan")), "published": float("nan"),
                          "fresh": float("nan")},
            "passes": not any(overlap)}


def main(argv=None):
    from frontierlab.pipeline.hf_eval import load_hf, suite
    from frontierlab.posttrain import gsm8k
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--model", default="smoke")
    s.add_argument("--revision", default=None)
    s.add_argument("--chat", action="store_true")
    s.add_argument("--smoke", action="store_true")
    s.add_argument("--n-gsm8k", type=int, default=500)
    s.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("new")
    c.add_argument("--guard", type=float, default=0.02)
    c.add_argument("--lambada-guard", type=float, default=0.05)
    a = ap.parse_args(argv)
    if a.cmd == "score":
        model, tok = load_hf(a.model, a.revision, a.smoke)
        res = suite(model, tok, chat=a.chat, n_gsm8k=a.n_gsm8k, smoke=a.smoke)
        qs = [f"{i}+1" for i in range(6)] if a.smoke else [q["question"] for q in gsm8k.load("test")[:a.n_gsm8k]]
        res = attach(res, contamination_section(model, tok, qs, a.smoke))
        res["model"] = {"name": a.model, "revision": a.revision}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res))
        print(json.dumps(res["summary"], indent=1))
    else:
        base, new = json.loads(Path(a.base).read_text()), json.loads(Path(a.new).read_text())
        print(report(compare(base, new, guards={"lambada_logprob": a.lambada_guard}, default_guard=a.guard)))


if __name__ == "__main__":
    main()
