import math

import numpy as np
import torch

from frontierlab.evals import suite_v1 as ref
from frontierlab.labkit import load_target
from frontierlab.model import LM, toy

lab = load_target(__file__)

VOCAB = ref.TaskVocab(keys=list(range(100, 140)), values=list(range(200, 240)), remember=[10, 11], ignore=[12, 11],
                      period=[13], filler=np.random.default_rng(1).integers(20, 60, size=20000))


def test_oracle_matches_reference_and_ignores_decoys():
    for hops, hard in ((1, 0), (1, 2), (2, 0), (3, 0)):
        for it in ref.make_items(VOCAB, 256, 15, hops=hops, depth=None, distractors=3, hard=hard, seed=hops):
            kw = dict(remember=VOCAB.remember, keys=set(VOCAB.keys), candidates=it["candidates"])
            assert lab.oracle_answer(it["ids"], **kw) == it["answer"]
            assert lab.oracle_answer(it["ids_ablated"], **kw) is None
    # a decoy with another prefix, placed first, must not be followed
    ids = [12, 11, 101, 230, 13] + [10, 11, 101, 205, 13] + [10, 11, 101]
    assert lab.oracle_answer(ids, [10, 11], {101}, [205, 230]) == 205
    # a loop never reaches a candidate
    assert lab.oracle_answer([10, 11, 101, 102, 13, 10, 11, 102, 101, 13, 10, 11, 101], [10, 11], {101, 102}, [205]) is None


def test_candidate_score():
    logits = torch.tensor([0.0, 2.0, 1.0, 3.0, -1.0])
    ok, lp = lab.candidate_score(logits, [1, 2, 4], 1)
    assert ok and abs(lp - (2.0 - math.log(math.exp(2) + math.exp(1) + math.exp(-1)))) < 1e-9
    ok, _ = lab.candidate_score(logits, [1, 3], 1)
    assert not ok
    ok, lp = lab.candidate_score(torch.zeros(5), [0, 1, 2, 3], 2)        # a tie is not correct
    assert not ok and abs(lp - math.log(0.25)) < 1e-9
    torch.manual_seed(0)
    model = LM(toy(vocab_size=256)).eval()
    items = ref.make_items(VOCAB, 96, 12, distractors=3, seed=0)
    expected = ref.score_items(model, items)
    with torch.no_grad():
        last = model(torch.tensor([it["ids"] for it in items])).logits[:, -1]
    for e, it, row in zip(expected, items, last):
        ok, lp = lab.candidate_score(row, it["candidates"], it["answer"])
        assert ok == e["correct"] and abs(lp - e["logp_cand"]) < 1e-5


def test_context_gain_matches_reference():
    torch.manual_seed(0)
    model = LM(toy(vocab_size=64)).eval()
    windows = torch.randint(0, 64, (3, 96))
    for W, lo, hi in ((16, 48, 96), (32, 32, 64), (8, 80, 96)):
        g = lab.context_gain(model, windows, W, lo, hi)
        with torch.no_grad():
            want = []
            for x in windows:
                full = model(x[None, :hi], labels=x[None, :hi]).per_token_loss[0]
                cut = model(x[None, lo - W:hi], labels=x[None, lo - W:hi]).per_token_loss[0]
                want.append(cut[W - 1:].mean() - full[lo - 1:hi - 1].mean())
        assert g.shape == (3,) and torch.allclose(g, torch.stack(want), atol=1e-5)
    assert lab.context_gain(model, windows, 40, 40, 96).abs().max() < 1e-5   # W = lo: nothing was cut


def test_effective_length_and_items_needed():
    cells = [{"length": 1024, "acc": (0.5, 0.3, 0.6)}, {"length": 256, "acc": (0.9, 0.8, 0.95)},
             {"length": 512, "acc": (0.8, 0.7, 0.9)}, {"length": 2048, "acc": (0.9, 0.75, 0.95)}]
    for t in (0.65, 0.75, 0.85, 0.2):
        assert lab.effective_length(cells, t) == ref.effective_length(cells, t)
    assert lab.effective_length(cells, 0.65) == 512
    assert lab.items_needed(0.35, 0.25) == 157 and lab.items_needed(0.5, 0.25) == 26
    assert lab.items_needed(0.9, 0.25) < 10
