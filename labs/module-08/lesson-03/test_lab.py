import torch

from frontierlab.labkit import load_target
from frontierlab.precision import E2M1, apply_rht, hadamard, qdq, round_to_format
from frontierlab.precision.hadamard import random_signs
from frontierlab.precision.linear import QLinear, get_recipe

lab = load_target(__file__)


def test_sr_round_e2m1_neighbours_and_unbiased():
    g = torch.Generator().manual_seed(0)
    x = (torch.rand(100_000, generator=g) * 14 - 7).double()
    u = torch.rand(100_000, generator=g).double()
    r = lab.sr_round_e2m1(x, u)
    xs = x.clamp(-6, 6)
    lo = round_to_format(xs, E2M1, "trunc")                 # neighbour toward zero
    assert bool(((r == lo) | ((r - xs).abs() <= 2.0)).all())
    assert set(torch.unique(r.abs()).tolist()) <= {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    on = torch.tensor([0.0, 1.5, -4.0, 6.0], dtype=torch.float64)
    assert torch.equal(lab.sr_round_e2m1(on, torch.full((4,), 0.999, dtype=torch.float64)), on)
    y = torch.full((200_000,), 2.6, dtype=torch.float64)
    assert abs(lab.sr_round_e2m1(y, torch.rand(200_000, generator=g).double()).mean().item() - 2.6) < 0.01


def test_rht_matches_reference_and_is_orthogonal():
    assert torch.allclose(lab.hadamard16(), hadamard(16), atol=1e-15)
    signs = random_signs(16, seed=0)
    x = torch.randn(5, 64, dtype=torch.float64)
    assert torch.allclose(lab.rht_blocks(x, signs), apply_rht(x, 16, 0), atol=1e-12)
    g = torch.randn(7, 64, dtype=torch.float64)
    assert torch.allclose(lab.rht_blocks(g, signs) @ lab.rht_blocks(x, signs).t(), g @ x.t(), atol=1e-10)


def test_int4_group_matches_reference():
    torch.manual_seed(0)
    w = torch.randn(48, 128) * 0.02
    assert torch.allclose(lab.int4_group_qdq(w), qdq(w, "int4-g32"), atol=1e-8, rtol=0)


def test_qat_backward_matches_the_qat_layer():
    torch.manual_seed(1)
    lin = torch.nn.Linear(64, 32, bias=False).double()
    q = QLinear.from_linear(lin, get_recipe("int4-qat"))
    x = torch.randn(40, 64, dtype=torch.float64, requires_grad=True)
    gy = torch.randn(40, 32, dtype=torch.float64)
    dx_ref, dw_ref = torch.autograd.grad(q(x), (x, q.weight), gy)
    dx, dw = lab.qat_backward(x.detach(), lin.weight.detach(), gy)
    assert torch.allclose(dx, dx_ref, atol=1e-10) and torch.allclose(dw, dw_ref, atol=1e-10)
