import torch

from frontierlab.labkit import load_target
from frontierlab.precision import QuantSpec, get_recipe, qdq
from frontierlab.precision.accum import limited_dot
from frontierlab.precision.linear import QLinear

lab = load_target(__file__)


def test_tile_and_block_quantisation_match_reference():
    torch.manual_seed(0)
    x = torch.randn(16, 384) * torch.rand(16, 1) * 30
    assert torch.allclose(lab.tile_qdq(x), qdq(x, QuantSpec("e4m3", (1, 128), "fp32")), atol=1e-6, rtol=0)
    w = torch.randn(256, 384) * 0.02
    w[3, 300] = 5.0
    ref = qdq(w, QuantSpec("e4m3", (128, 128), "fp32"))
    assert torch.allclose(lab.block_qdq(w), ref, atol=1e-7, rtol=0)
    assert torch.equal(lab.block_qdq(w.t().contiguous()).t(), lab.block_qdq(w))   # square blocks: W and W^T agree


def test_promoted_sum_matches_model_and_bounds_error():
    g = torch.Generator().manual_seed(0)
    p = torch.rand(8, 2048, generator=g, dtype=torch.float64)
    ones = torch.ones_like(p)
    for n_c in (None, 128):
        assert torch.equal(lab.promoted_sum(p, 13, n_c), limited_dot(p, ones, 13, n_c, "trunc"))
    exact = p.sum(-1)
    e_plain = ((lab.promoted_sum(p, 13, None) - exact).abs() / exact).max()
    e_prom = ((lab.promoted_sum(p, 13, 128) - exact).abs() / exact).max()
    assert e_prom < 0.2 * e_plain


def test_fp8_backward_matches_the_emulated_layer():
    torch.manual_seed(1)
    x = torch.randn(256, 384, dtype=torch.float64, requires_grad=True)
    lin = torch.nn.Linear(384, 128, bias=False).double()
    q = QLinear.from_linear(lin, get_recipe("fp8-deepseek"))
    gy = torch.randn(256, 128, dtype=torch.float64)
    ref_dx, ref_dw = torch.autograd.grad(q(x), (x, q.weight), gy)
    dx, dw = lab.fp8_backward(x.detach(), lin.weight.detach(), gy)
    assert torch.allclose(dx, ref_dx, atol=1e-10) and torch.allclose(dw, ref_dw, atol=1e-10)


def test_decide():
    assert lab.decide((0.01, 0.03), 0.02, (1.2, 1.4)) == "inconclusive"
    assert lab.decide((0.025, 0.03), 0.02, (1.2, 1.4)) == "reject"
    assert lab.decide((-0.01, 0.015), 0.02, (1.2, 1.4)) == "adopt"
    assert lab.decide((-0.01, 0.015), 0.02, (0.95, 1.1)) == "reject"
    assert lab.decide((-0.01, 0.015), 0.02, None) == "numerics ok, speed not measured"
    assert lab.decide((-0.01, 0.015), 0.02, None, per_seed=[0.03, -0.03]) == "inconclusive"
    assert lab.decide((-0.01, 0.015), 0.02, (1.2, 1.4), per_seed=[0.01, -0.005]) == "adopt"
    assert lab.decide((0.025, 0.03), 0.02, None, per_seed=[0.03, 0.03]) == "reject"
