"""A colleague's refactor of three pipeline pieces. Their message:

    "I cleaned up the pair builder, the distillation loss and the example encoder so they are shared by
    every stage. The pipeline runs end to end without errors. DPO's reward accuracy looks fine, the
    distillation stage's loss goes down, but the final model is worse than the SFT start and it
    sometimes never stops writing. Probably just needs more steps?"

    python labs/module-13/project/buggy_pipeline.py            # their pieces: a short SFT -> DPO -> distil run
    python labs/module-13/project/buggy_pipeline.py --fixed    # the reference pieces, same run
    PIPE_HOOKS=buggy pytest labs/module-13/project             # the piece tests against their versions

There are three bugs. Find each from its symptom, name the test or log line that isolates it, then fix it.
"""

from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch

from frontierlab.pipeline import distill as DI
from frontierlab.pipeline import dpo as D
from frontierlab.pipeline import toy
from frontierlab.pipeline.seqs import Example, train_sft
from frontierlab.posttrain.policy import masked_mean
from frontierlab.posttrain.tokenizer import BOS, EOS, TOK


def make_pair(scores):
    s = np.asarray(scores, dtype=float)
    if s.max() == s.min():
        return None
    order = np.argsort(s)                       # "sorted ascending, so the first is the best"
    return int(order[0]), int(order[-1])


def distill_advantage(logp, teacher_logp):
    return (logp - teacher_logp).detach()       # "advantage = the reverse KL"


def sft_example(problem, text, finished=True):
    return Example.of([BOS] + TOK.encode(problem.prompt), TOK.encode(text))   # "EOS is added by the loader"


def run(pieces, steps: int = 120, seed: int = 0) -> dict:
    """A short SFT -> DPO -> distillation run on the toy world with the given pieces; returns the logs."""
    from frontierlab.posttrain.sft import ensure_sft, load_policy
    torch.manual_seed(seed)
    base = load_policy(ensure_sft("runs/m12/sft"))
    teacher = load_policy(DI.ensure_teacher("runs/m13/teacher"))
    out = {"sft-start": toy.eval_v2(base)["summary"]}
    # preference stage
    probs = toy.train_problems(4000, seed=11)
    cands = toy.sample_texts(base, probs, 4, 1.0, seed=3)
    pairs = []
    for p, cs in zip(probs, cands):
        pool = list(dict.fromkeys(cs))
        pick = pieces.make_pair([toy.aspect_score(p, t, f) for t, f in pool])
        if pick:
            pairs.append((pieces.sft_example(p, *pool[pick[0]]), pieces.sft_example(p, *pool[pick[1]])))
    pol = copy.deepcopy(base)
    tr = D.train_dpo(pol, base, pairs, steps, beta=0.1, nll_coef=0.2, lr=5e-5, seed=seed)
    out["dpo"] = {**toy.eval_v2(pol)["summary"], "reward_acc": tr["final"]["reward_acc"]}
    # distillation stage (on-policy, the colleague's advantage)
    fn = lambda logp, tl, mask, sl, tlog: -masked_mean(torch.exp(logp - logp.detach()) * pieces.distill_advantage(logp, tl), mask)
    d = DI.train_opd(pol, teacher, lambda st: toy.train_problems(16, seed=900 + st, ops="+", tags="."), steps,
                     lr=3e-4, seed=seed, loss_fn=fn, log_every=steps // 4)
    out["distill"] = {**toy.eval_v2(pol)["summary"], "rkl_first": d["history"][0]["rkl_sampled"],
                      "rkl_last": d["history"][-1]["rkl_sampled"], "loss_last": d["history"][-1]["loss"]}
    # an SFT top-up on teacher text with the colleague's example encoder
    ex = [pieces.sft_example(p, t, f) for p, ts in zip(probs[:2000], toy.sample_texts(teacher, probs[:2000], 1))
          for t, f in ts]
    train_sft(pol, ex, steps, batch=64, lr=3e-4, seed=seed)
    from frontierlab.evals.suite_v2.toy import greedy_responses
    from frontierlab.posttrain.tasks import problems_from, split_problems
    _, held = split_problems(2)
    hp = problems_from(held, ".", 2, "+")[:200]
    resp = greedy_responses(pol, hp, 8)
    out["final"] = {**toy.eval_v2(pol)["summary"], "no_eos_rate": float((~(resp == EOS).any(-1)).float().mean())}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixed", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(8)
    import importlib.util
    from pathlib import Path
    if a.fixed:
        spec = importlib.util.spec_from_file_location("pieces", Path(__file__).with_name("pieces.py"))
        pieces = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pieces)
    else:
        import sys
        pieces = sys.modules[__name__]
    t0 = time.perf_counter()
    out = run(pieces)
    for stage, v in out.items():
        print(f"{stage:10s} " + "  ".join(f"{k} {x:.3f}" for k, x in v.items()))
    print(f"{time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
