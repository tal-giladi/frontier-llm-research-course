import math

import torch

from frontierlab.labkit import load_target
from frontierlab.model import LM, toy
from frontierlab.optim import cost
from frontierlab.optim.muon import newton_schulz as ref_ns, orthogonal_polar, split_params

lab = load_target(__file__)


def test_ns_step_is_the_quintic_on_singular_values():
    s = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    out = lab.ns_step(torch.diag(s), *lab.QUINTIC)
    a, b, c = lab.QUINTIC
    assert torch.allclose(out.diagonal(), a * s + b * s ** 3 + c * s ** 5, atol=1e-14)


def test_newton_schulz_matches_reference_for_every_orientation():
    torch.manual_seed(0)
    for shape in ((16, 40), (40, 16), (24, 24)):
        G = torch.randn(*shape, dtype=torch.float64)
        out = lab.newton_schulz(G)
        assert out.shape == G.shape
        assert (out - ref_ns(G, "quintic5", dtype=torch.float64)).abs().max() < 1e-12, shape


def test_orthogonality_band_and_cubic_convergence():
    torch.manual_seed(1)
    G = torch.randn(32, 96, dtype=torch.float64)
    s = torch.linalg.svdvals(lab.newton_schulz(G))
    assert 0.6 < s.min() and s.max() < 1.25                       # quintic: near 1, not exactly 1
    O = lab.newton_schulz(G, coeffs=(1.5, -0.5, 0.0), steps=40)   # classic cubic converges to U V^T
    assert (O - orthogonal_polar(G)).abs().max() < 1e-10


def test_muon_direction_and_apply_match_shared_optimizer():
    from frontierlab.optim.muon import MuonAdamW
    torch.manual_seed(2)
    W0 = torch.randn(48, 32, dtype=torch.float64)
    grads = [torch.randn(48, 32, dtype=torch.float64) for _ in range(3)]
    for adjust in ("match_rms", "original"):
        W, buf = W0.clone(), torch.zeros_like(W0)
        p = torch.nn.Parameter(W0.clone())
        opt = MuonAdamW([{"params": [p], "kind": "muon", "lr": 0.02, "weight_decay": 0.1, "adjust": adjust,
                          "ns_dtype": "float64"}])
        for g in grads:
            O = lab.muon_direction(g, buf)
            lab.muon_apply(W, O, 0.02, 0.1, adjust)
            p.grad = g.clone()
            opt.step()
        assert (W - p.detach()).abs().max() < 1e-12, adjust


def test_muon_param_names():
    model = LM(toy(vocab_size=97))
    assert sorted(lab.muon_param_names(model)) == sorted(n for n, _ in split_params(model)["muon"])


def test_ns_flops():
    for shape in ((768, 768), (256, 768), (2816, 768)):
        assert lab.ns_flops(shape, 5) == cost.ns_flops(shape, 5)
    m = 768
    assert math.isclose(lab.ns_flops((m, m), 5), 30 * m ** 3)
