"""Validity checks for Eval v1's synthetic tasks, using YOUR lab.py functions on real items.

    python labs/module-04/lesson-01/check_eval.py --run runs/m04/base-cpu
    LAB_TARGET=solution python labs/module-04/lesson-01/check_eval.py --run runs/m04/base-cpu

Four checks, each printed PASS or FAIL:

1. Solvable: your oracle answers every item correctly from the context alone.
2. Not solvable without the evidence: on the evidence-ablated twin, the oracle finds nothing.
3. No prior leak: the model's accuracy on the ablated twins is consistent with chance (its 95% CI
   contains 1/K). If it is not, the answer can be guessed without reading the context.
4. Your candidate scorer agrees with the reference scorer on every item.

Then it prints the sample size you would need to tell the model's measured accuracy from chance.
"""

import argparse
import os
from pathlib import Path

import torch

from frontierlab.data.loader import TokenData
from frontierlab.data.prepare import DEFAULT_OUT
from frontierlab.evals import suite_v1 as v1
from frontierlab.labkit import load_target
from frontierlab.stats import bootstrap_ci

lab = load_target(str(Path(__file__).parent / "test_lab.py"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    print(f"checking {os.environ.get('LAB_TARGET', 'lab')}.py")
    vocab = v1.vocab_from_tokenizer(DEFAULT_OUT / "tokenizer.json", TokenData("val"))
    model = v1.load_model(a.run, a.device)
    items = (v1.make_items(vocab, a.length, a.n, hops=1, depth=None, distractors=3, hard=1, seed=11)
             + v1.make_items(vocab, a.length, a.n, hops=2, distractors=3, seed=12))
    kw = lambda it: dict(remember=vocab.remember, keys=set(vocab.keys), candidates=it["candidates"])  # noqa: E731
    solved = sum(lab.oracle_answer(it["ids"], **kw(it)) == it["answer"] for it in items)
    leaked = sum(lab.oracle_answer(it["ids_ablated"], **kw(it)) is not None for it in items)
    print(f"{'PASS' if solved == len(items) else 'FAIL'}  1. oracle solves {solved}/{len(items)} items")
    print(f"{'PASS' if leaked == 0 else 'FAIL'}  2. oracle finds evidence in {leaked}/{len(items)} ablated twins")
    ref, abl = v1.score_items(model, items, device=a.device), v1.score_items(model, items, "ids_ablated", device=a.device)
    chance = sum(1 / len(it["candidates"]) for it in items) / len(items)
    m, lo, hi = bootstrap_ci([float(s["correct"]) for s in abl], n_boot=4000)
    print(f"{'PASS' if lo <= chance <= hi else 'FAIL'}  3. ablated accuracy {m:.3f} [{lo:.3f}, {hi:.3f}], chance {chance:.3f}")
    agree = 0
    with torch.no_grad():
        for it, r in zip(items, ref):
            last = model(torch.tensor([it["ids"]], device=a.device)).logits[0, -1].float().cpu()
            ok, lp = lab.candidate_score(last, it["candidates"], it["answer"])
            agree += ok == r["correct"] and abs(lp - r["logp_cand"]) < 1e-4
    print(f"{'PASS' if agree == len(items) else 'FAIL'}  4. your scorer agrees with the reference on {agree}/{len(items)} items")
    acc = sum(s["correct"] for s in ref) / len(ref)
    print(f"\nmodel accuracy with the evidence: {acc:.3f} (chance {chance:.3f}) on {len(items)} items")
    if acc > chance:
        print(f"items needed to separate {acc:.3f} from chance with 80% power: {lab.items_needed(acc, chance)}")
    else:
        print("accuracy is not above chance: no item count makes this a detectable effect")


if __name__ == "__main__":
    main()
