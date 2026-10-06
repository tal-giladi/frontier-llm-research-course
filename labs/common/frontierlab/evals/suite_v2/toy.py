"""Eval Suite v2 for the toy arithmetic world (free CPU path).

    python -m frontierlab.evals.suite_v2.toy score runs/m12/sft/policy.pt --out runs/m12/eval/sft.json
    python -m frontierlab.evals.suite_v2.toy compare runs/m12/eval/sft.json runs/m12/eval/rl.json

Components (held-out triples only, fixed order, fixed sampling seed):

==========================  ===========  =====================================================
component                   kind         item score
==========================  ===========  =====================================================
``add_pass1``               task         fraction of n sampled responses that are correct (+, tag ``.``)
``add_greedy``              task         greedy response correct
``add_pass@k``              task         unbiased pass@k from the same n samples
``sub_greedy``              retention    greedy subtraction correct (a skill RL did not train)
``sft_nll``                 retention    minus the teacher-forced loss of the SFT target (all ops and tags)
``if_strict``               instruction  the greedy response obeys its tag (P, Q, E), any number
``if_correct``              instruction  obeys the tag *and* the number is right
==========================  ===========  =====================================================

Instruction-level and prompt-level accuracy coincide here (one instruction per prompt); the IFEval
subset for the main path (:mod:`.ifeval`) reports both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from frontierlab.evals.suite_v2.core import VERSION, compare, pass_at_k, report
from frontierlab.posttrain.policy import sample
from frontierlab.posttrain.sft import load_policy
from frontierlab.posttrain.tasks import (encode_prompts, encode_sft, follows_instruction, problems_from,
                                         response_text, score, split_problems)
from frontierlab.posttrain.tokenizer import EOS, PAD


@torch.no_grad()
def greedy_responses(model, problems, max_new: int = 8) -> torch.Tensor:
    was = model.training
    model.eval()
    prompts = encode_prompts(problems)
    cache = model.new_cache()
    logits = model(prompts, cache=cache).logits[:, -1]
    out = torch.full((len(problems), max_new), PAD, dtype=torch.long)
    done = torch.zeros(len(problems), dtype=torch.bool)
    for t in range(max_new):
        nxt = torch.where(done, torch.full((len(problems),), PAD), logits.argmax(-1))
        out[:, t] = nxt
        done |= nxt == EOS
        if bool(done.all()):
            break
        logits = model(nxt[:, None], cache=cache).logits[:, -1]
    model.train(was)
    return out


@torch.no_grad()
def run_suite(model, digits: int = 2, n_items: int = 200, n_samples: int = 8, k: int = 4, max_new: int = 8,
              temperature: float = 1.0, seed: int = 4321) -> dict:
    _, held = split_problems(digits)
    add = problems_from(held, ".", digits, "+")[:n_items]
    sub = problems_from(held, ".", digits, "-")[:n_items]
    tagged = [p for op in "+-" for p in problems_from(held, "PQE", digits, op)[:n_items]]
    allsft = [p for op in "+-" for p in problems_from(held, ".PQE", digits, op)[:n_items]]
    comps = {}
    # task
    g = torch.Generator().manual_seed(seed)
    rep = [p for p in add for _ in range(n_samples)]
    ro = sample(model, encode_prompts(rep), max_new, temperature, g)
    c = score(ro.response, rep).view(len(add), n_samples).sum(1)
    comps["add_pass1"] = {"kind": "task", "items": (c / n_samples).tolist()}
    comps[f"add_pass@{k}"] = {"kind": "task", "items": [pass_at_k(n_samples, int(ci), k) for ci in c]}
    comps["add_greedy"] = {"kind": "task", "items": score(greedy_responses(model, add, max_new), add).tolist()}
    # retention
    comps["sub_greedy"] = {"kind": "retention",
                           "items": score(greedy_responses(model, sub, max_new), sub).tolist()}
    nll = []
    for i in range(0, len(allsft), 256):
        ids, mask = encode_sft(allsft[i:i + 256])
        logits = model(ids).logits[:, :-1].float()
        tok = torch.nn.functional.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction="none")
        m = mask[:, 1:].float()
        nll.extend((-(tok * m).sum(1) / m.sum(1)).tolist())
    comps["sft_nll"] = {"kind": "retention", "items": nll}
    # instruction following
    resp = greedy_responses(model, tagged, max_new)
    strict, correct = [], []
    for row, p in zip(resp, tagged):
        text, fin = response_text(row)
        f = follows_instruction(text, fin, p)
        strict.append(float(f))
        correct.append(float(f and text == p.target))
    comps["if_strict"] = {"kind": "instruction", "items": strict}
    comps["if_correct"] = {"kind": "instruction", "items": correct}
    pins = {"digits": digits, "n_items": n_items, "n_samples": n_samples, "k": k, "max_new": max_new,
            "temperature": temperature, "seed": seed, "split_seed": 1234}
    return {"version": VERSION, "pins": pins, "components": comps,
            "summary": {k_: sum(v["items"]) / len(v["items"]) for k_, v in comps.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("policy")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--n-items", type=int, default=200)
    s.add_argument("--n-samples", type=int, default=8)
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("new")
    c.add_argument("--guard", type=float, default=0.02)
    a = ap.parse_args(argv)
    if a.cmd == "score":
        res = run_suite(load_policy(a.policy), n_items=a.n_items, n_samples=a.n_samples)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res))
        print(json.dumps(res["summary"], indent=2))
    else:
        base, new = json.loads(Path(a.base).read_text()), json.loads(Path(a.new).read_text())
        print(report(compare(base, new, default_guard=a.guard)))


if __name__ == "__main__":
    main()
