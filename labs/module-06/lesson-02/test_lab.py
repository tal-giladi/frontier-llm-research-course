import torch

from frontierlab.blocks import BlockLM, with_blocks
from frontierlab.blocks.hyperconn import ManifoldHC, amax_gain, doubly_stochastic_error, residual_io_elements, sinkhorn_knopp
from frontierlab.labkit import load_target
from frontierlab.model import toy

lab = load_target(__file__)


def test_sinkhorn_matches_and_is_doubly_stochastic():
    torch.manual_seed(0)
    x = torch.randn(32, 4, 4, dtype=torch.float64) * 2
    for t in (1, 5, 20):
        assert torch.allclose(lab.sinkhorn(x, t), sinkhorn_knopp(x, t), atol=1e-14)
    assert doubly_stochastic_error(lab.sinkhorn(x, 300)) < 1e-12
    big = torch.full((1, 3, 3), 800.0, dtype=torch.float64)                 # exp(800) overflows without the shift
    assert torch.isfinite(lab.sinkhorn(big, 5)).all()


def test_mhc_mix_matches_module():
    torch.manual_seed(0)
    mod = ManifoldHC(4, 16, 1).double()
    X = torch.randn(2, 3, 4, 16, dtype=torch.float64)
    lin = torch.nn.Linear(16, 16).double()
    pre, post, res = mod.coefficients(X)
    assert (lab.mhc_mix(X, lin, pre, post, res) - mod(X, lin)).abs().max().item() < 1e-12


def test_composite_gain_matches_reference():
    torch.manual_seed(0)
    mats = [torch.randn(2, 3, 4, 4, dtype=torch.float64) * 0.6 for _ in range(5)]
    ref = amax_gain(mats)
    fwd, bwd = lab.composite_gain(mats)
    assert abs(fwd - ref["composite_fwd"]) < 1e-12 and abs(bwd - ref["composite_bwd"]) < 1e-12
    m = BlockLM(with_blocks(toy(vocab_size=50), residual="mhc"))
    m.track_hyper = True
    m(torch.randint(0, 50, (1, 8)))
    res = [hy[k].last_res for hy in m.model.hyper for k in ("attn", "mlp")]
    f, b = lab.composite_gain(res)
    assert abs(f - 1) < 1e-4 and abs(b - 1) < 1e-3                        # doubly stochastic: gain 1


def test_residual_io():
    for n, C in ((1, 768), (4, 768), (4, 2560), (8, 128)):
        kind = "plain" if n == 1 else "hc"
        assert lab.residual_io(n, C) == residual_io_elements(n, C, kind)
