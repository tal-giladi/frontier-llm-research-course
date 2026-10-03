import torch

from frontierlab.labkit import load_target
from frontierlab.model import LM, toy
from frontierlab.perf.chunked_ce import lm_loss_chunked

lab = load_target(__file__)


def test_bench_warms_up_syncs_and_repeats():
    calls, syncs = [], []
    out = lab.bench(lambda: calls.append(len(syncs)), warmup=2, repeats=5, sync=lambda: syncs.append(1))
    assert len(out) == 5 and len(calls) == 7 and all(t >= 0 for t in out)
    # every timed call is bracketed by a sync before and after
    assert len(syncs) >= 10
    timed_calls = calls[2:]
    assert all(b - a >= 2 for a, b in zip(timed_calls, timed_calls[1:]))


def test_grad_agreement_detects_equal_and_different_losses():
    torch.manual_seed(0)
    m = LM(toy(vocab_size=101)).double()
    x = torch.randint(0, 101, (2, 16))
    plain = lambda mm, i: mm(i, labels=i).loss                     # noqa: E731
    same = lab.grad_agreement(m, x, plain, lambda mm, i: lm_loss_chunked(mm, i, 8))
    assert same["loss_diff"] < 1e-6 and same["max_grad_diff"] < 1e-6
    wrong = lab.grad_agreement(m, x, plain, lambda mm, i: 1.1 * mm(i, labels=i).loss)
    assert wrong["loss_diff"] > 0.1 and wrong["max_grad_diff"] > 1e-4
    assert all(p.grad is None for p in m.parameters())               # nothing accumulated


def test_decide():
    assert lab.decide(False, (1.5, 2.0), 0.1) == "reject"
    assert lab.decide(True, (0.80, 0.90), 0.1) == "reject"
    assert lab.decide(True, (0.97, 1.10), 0.5) == "adopt"
    assert lab.decide(True, (0.90, 1.10), 0.5) == "inconclusive"
    assert lab.decide(True, (1.10, 1.30), 0.95) == "adopt"                # clearly faster, same memory
    assert lab.decide(True, (0.97, 1.20), 0.95) == "inconclusive"          # neither smaller nor clearly faster
