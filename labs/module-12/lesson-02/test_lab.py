import torch

from frontierlab.labkit import load_target
from frontierlab.posttrain import advantages as A
from frontierlab.posttrain import kl as K

lab = load_target(__file__)


def test_group_advantages_match_reference():
    torch.manual_seed(0)
    r = (torch.rand(6, 8) < 0.4).float()
    r[0] = 1.0                                             # a zero-variance group
    for baseline in ("none", "mean", "loo"):
        for scale in ("none", "group", "batch"):
            got = lab.group_advantages(r, baseline, scale)
            assert got.dtype == r.dtype
            assert torch.allclose(got, A.group_advantages(r, baseline, scale), atol=1e-6), (baseline, scale)


def test_worked_example_numbers():
    r = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(lab.group_advantages(r, "mean", "none"), torch.tensor([[0.75, -0.25, -0.25, -0.25]]))
    assert torch.allclose(lab.group_advantages(r, "loo", "none"), torch.tensor([[1.0, -1 / 3, -1 / 3, -1 / 3]]))
    assert torch.allclose(lab.group_advantages(r, "mean", "group"), torch.tensor([[1.5, -0.5, -0.5, -0.5]]),
                          atol=1e-5)


def test_zero_variance():
    r = torch.tensor([[1.0, 1, 1], [0.0, 0, 0], [0.0, 1, 0]])
    assert lab.zero_variance(r).tolist() == [True, True, False]


def test_kl_estimators_and_gradients():
    lp, ref = torch.log(torch.tensor([0.5, 0.25])).requires_grad_(True), torch.log(torch.tensor([0.25, 0.5]))
    got, want = lab.kl_estimators(lp, ref), K.estimators(lp, ref)
    for k in ("k1", "k2", "k3"):
        assert torch.allclose(got[k], want[k])
    got["k3"].sum().backward()
    assert lp.grad is not None
    torch.manual_seed(1)
    logits, ref_logits = torch.randn(6), torch.randn(6)
    exact = K.exact_categorical(logits, ref_logits)
    for k in ("k1", "k2", "k3"):
        assert torch.allclose(lab.expected_loss_gradient(logits, ref_logits, k), exact[k], atol=1e-12)


def test_entropy():
    assert abs(lab.entropy(torch.zeros(4)).item() - torch.log(torch.tensor(4.0)).item()) < 1e-6
    x = torch.randn(3, 5, 7)
    p = torch.softmax(x, -1)
    assert torch.allclose(lab.entropy(x), -(p * p.log()).sum(-1), atol=1e-5)


def test_decide():
    assert lab.decide(0.1, 0.02, 0.2) == "better"
    assert lab.decide(-0.1, -0.2, -0.01) == "worse"
    assert lab.decide(0.1, -0.02, 0.2) == "inconclusive"
