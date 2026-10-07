"""One test per pipeline piece. Default: the reference pieces (pieces.py); PIPE_HOOKS=buggy checks the colleague's."""

import importlib.util
import os
from pathlib import Path

import torch

from frontierlab.pipeline import distill as DI
from frontierlab.posttrain.tasks import Problem
from frontierlab.posttrain.tokenizer import EOS, TOK

HERE = Path(__file__).resolve().parent
_name = "buggy_pipeline.py" if os.environ.get("PIPE_HOOKS") == "buggy" else "pieces.py"
_spec = importlib.util.spec_from_file_location(f"m13_project_{_name[:-3]}", HERE / _name)
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


def test_pair_chosen_is_the_best_scored():
    assert P.make_pair([0.5, 1.0, 0.0, 0.5]) == (1, 2)
    assert P.make_pair([0.3, 0.3]) is None


def test_distillation_advantage_lowers_reverse_kl():
    g = torch.Generator().manual_seed(0)
    s = torch.randn(6, generator=g, dtype=torch.float64).requires_grad_(True)
    t = torch.randn(6, generator=g, dtype=torch.float64)
    ls, lt = torch.log_softmax(s, -1), torch.log_softmax(t, -1)
    grad = torch.zeros(6, dtype=torch.float64)
    for y in range(6):
        adv = P.distill_advantage(ls[y], lt[y])
        gy, = torch.autograd.grad(-(adv * ls[y]), s, retain_graph=True)
        grad += ls[y].exp().detach() * gy
    true, = torch.autograd.grad(DI.exact_reverse_kl(s[None, None], t[None, None]).sum(), s)
    assert torch.allclose(grad, true, atol=1e-12)     # descending this loss descends the reverse KL


def test_finished_examples_end_with_eos():
    p = Problem(".", 7, "+", 35, 2)
    ex = P.sft_example(p, "42")
    assert ex.response[-1] == EOS and TOK.decode(list(ex.response)) == "42"
    assert P.sft_example(p, "42", finished=False).response[-1] != EOS
