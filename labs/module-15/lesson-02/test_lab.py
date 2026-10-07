import math

import torch

from frontierlab.labkit import load_target
from frontierlab.ttc import speculative as SP

lab = load_target(__file__)


def test_residual_by_hand():
    p = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    q = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    assert torch.allclose(lab.residual(p, q), torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    assert torch.allclose(lab.residual(p, p), p)
    g = torch.Generator().manual_seed(0)
    P = torch.softmax(torch.randn(4, 7, generator=g, dtype=torch.float64), -1)
    Q = torch.softmax(torch.randn(4, 7, generator=g, dtype=torch.float64), -1)
    assert torch.allclose(lab.residual(P, Q), SP.residual(P, Q))


def test_accept_reject_deterministic_cases():
    V = 3
    p = torch.tensor([[[0.5, 0.3, 0.2], [0.1, 0.1, 0.8], [0.0, 1.0, 0.0]]], dtype=torch.float64)
    q = torch.tensor([[[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]]], dtype=torch.float64)
    # draft 0 is token 0 (p/q = 2.5, always accepted), draft 1 is token 2 (p/q = 1): both accepted -> bonus from p[2]
    n, nxt = lab.accept_reject(p, q, torch.tensor([[0, 2]]), torch.tensor([[0.99, 0.99]], dtype=torch.float64))
    assert int(n) == 2 and int(nxt) == 1
    # draft 0 is token 1: p/q = 0.6; u = 0.7 rejects it -> residual of position 0 is all on token 0
    n, nxt = lab.accept_reject(p, q, torch.tensor([[1, 2]]), torch.tensor([[0.7, 0.0]], dtype=torch.float64))
    assert int(n) == 0 and int(nxt) == 0
    # a later acceptance does not count after a rejection
    n, _ = lab.accept_reject(p, q, torch.tensor([[1, 2]]), torch.tensor([[0.7, 0.0]], dtype=torch.float64))
    assert int(n) == 0


def test_accept_reject_emits_the_target_distribution():
    g = torch.Generator().manual_seed(1)
    V, B = 6, 200_000
    p = torch.softmax(torch.randn(V, generator=g, dtype=torch.float64) * 1.5, -1)
    q = torch.softmax(torch.randn(V, generator=g, dtype=torch.float64) * 1.5, -1)
    d = torch.multinomial(q.expand(B, V), 1, replacement=True, generator=g)
    u = torch.rand(B, 1, generator=g, dtype=torch.float64)
    n, nxt = lab.accept_reject(torch.stack([p, p]).expand(B, 2, V), q.expand(B, 1, V), d, u, g)
    first = torch.where(n == 1, d[:, 0], nxt)
    emp = torch.bincount(first, minlength=V).double() / B
    assert (emp - p).abs().max() < 0.005
    assert abs(n.double().mean() - torch.minimum(p, q).sum()) < 0.005


def test_speed_formulas():
    assert lab.expected_tokens_per_round(0.0, 5) == 1.0
    assert lab.expected_tokens_per_round(1.0, 3) == 4.0
    assert math.isclose(lab.expected_tokens_per_round(0.7, 4), 1 + 0.7 + 0.49 + 0.343 + 0.2401)
    assert math.isclose(lab.walltime_improvement(0.7, 4, 0.1), SP.walltime_improvement(0.7, 4, 0.1))
