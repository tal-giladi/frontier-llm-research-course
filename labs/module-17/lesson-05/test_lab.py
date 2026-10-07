import torch

from frontierlab.interp import sparse as SPR
from frontierlab.labkit import load_target
from frontierlab.model import LM

lab = load_target(__file__)


def test_magnitude_mask_by_hand_and_against_course():
    W = torch.tensor([[0.1, -3.0], [2.0, 0.5]])
    assert torch.equal(lab.magnitude_mask(W, 0.5), torch.tensor([[False, True], [True, False]]))
    assert int(lab.magnitude_mask(W, 0.01).sum()) == 1
    torch.manual_seed(0)
    m = LM(SPR.task_config())
    ref = {n: p.detach().clone() for n, p in m.named_parameters()}
    SPR.apply_topk_(m, 0.2)
    for n, p in m.named_parameters():
        if p.dim() == 2 and "layers" in n:
            assert torch.equal(p != 0, lab.magnitude_mask(ref[n], 0.2))


def test_density_schedule():
    assert lab.density_at(0, 100, 0.1) == 1.0
    assert abs(lab.density_at(25, 100, 0.1) - 0.55) < 1e-12
    assert abs(lab.density_at(50, 100, 0.1) - 0.1) < 1e-12 and abs(lab.density_at(99, 100, 0.1) - 0.1) < 1e-12


def test_gated_mean_ablation():
    a = torch.tensor([[1.0, 2.0, 3.0]])
    mu = torch.tensor([10.0, 20.0, 30.0])
    assert torch.equal(lab.gated(a, torch.tensor([1.0, 0.0, 0.5]), mu), torch.tensor([[1.0, 20.0, 16.5]]))


def test_rates():
    g = [{"claims": True, "names": True}, {"claims": True, "names": False}, {"claims": False, "names": True},
         {"claims": False, "names": False}]
    assert lab.rates(g) == {"claims": 0.5, "correct": 0.25}
