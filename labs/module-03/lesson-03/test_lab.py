import math

import torch

from frontierlab.attention import ops, probes
from frontierlab.labkit import load_target

lab = load_target(__file__)


def test_sink_softmax_by_hand_and_against_reference():
    p = lab.sink_softmax(torch.zeros(1, 1, 1, 2), torch.tensor([0.0]))
    assert torch.allclose(p, torch.full((1, 1, 1, 2), 1 / 3))                    # e^0 / (e^0 + e^0 + e^0)
    p = lab.sink_softmax(torch.tensor([[[[math.log(3.0), 0.0]]]]), torch.tensor([0.0]))
    assert torch.allclose(p, torch.tensor([[[[0.6, 0.2]]]]))                        # 3/5, 1/5; sink 1/5
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 3, 5, 5, generator=g, dtype=torch.float64) * 4
    logits = logits.masked_fill(~ops.band_mask(torch.arange(5), torch.arange(5)), float("-inf"))
    sink = torch.randn(3, generator=g, dtype=torch.float64)
    assert torch.allclose(lab.sink_softmax(logits, sink), ops.sink_softmax(logits, sink), atol=1e-14)


def test_sink_softmax_does_not_overflow():
    p = lab.sink_softmax(torch.tensor([[[[1000.0, 999.0]]]]), torch.tensor([-1000.0]))
    assert torch.isfinite(p).all() and abs(p.sum().item() - 1) < 1e-6


def test_first_token_mass_and_max_logit():
    probs = torch.zeros(1, 1, 4, 4)
    probs[0, 0, :, 0] = torch.tensor([1.0, 0.5, 0.2, 0.1])
    assert abs(lab.first_token_mass(probs, skip=2) - 0.15) < 1e-7
    assert lab.first_token_mass(probs, skip=2) == probes.first_token_mass(probs, skip=2)
    lg = torch.tensor([[[[1.0, float("-inf")], [3.0, 2.0]]]])
    assert lab.max_logit(lg) == 3.0


def test_massive_ratio():
    h = torch.ones(1, 3, 4)
    h[0, 0, 2] = -500.0
    assert lab.massive_ratio(h) == 500.0
    g = torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(1))
    assert abs(lab.massive_ratio(g) - probes.massive_ratio(g)) < 1e-6
