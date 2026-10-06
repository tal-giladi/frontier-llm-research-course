import torch

from frontierlab.labkit import load_target
from frontierlab.posttrain import losses as Lo
from frontierlab.posttrain.policy import response_mask
from frontierlab.posttrain.tokenizer import EOS, PAD

lab = load_target(__file__)


def test_response_mask():
    r = torch.tensor([[5, 6, EOS, PAD, PAD], [EOS, PAD, PAD, PAD, PAD], [5, 6, 7, 8, 9], [5, EOS, 7, EOS, 9]])
    m, f = lab.response_mask(r)
    rm, rf = response_mask(r)
    assert torch.equal(m.float(), rm) and torch.equal(f, rf)
    assert m[1].tolist() == [1, 0, 0, 0, 0]          # an immediate EOS is still one response token


def test_aggregate_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(8, 6)
    lengths = torch.tensor([1, 6, 3, 2, 6, 4, 5, 1])
    m = (torch.arange(6)[None] < lengths[:, None]).float()
    for mode in Lo.AGGREGATIONS:
        got = lab.aggregate(x, m, mode, group_size=4, norm_len=6)
        assert torch.allclose(got, Lo.aggregate(x, m, mode, group_size=4, norm_len=6), atol=1e-6), mode


def test_length_bias_weights():
    m = torch.tensor([[1.0, 1, 0, 0], [1.0, 1, 1, 1]])
    pt = torch.zeros(2, 4, requires_grad=True)
    lab.aggregate(pt, m, "seq_mean_token_mean").backward()
    assert torch.allclose(pt.grad, torch.tensor([[0.25, 0.25, 0, 0], [0.125] * 4]))


def test_ratios_and_clipping():
    torch.manual_seed(0)
    logp, old = torch.randn(3, 4, requires_grad=True), torch.randn(3, 4)
    m = torch.tensor([[1.0, 1, 1, 0], [1.0, 0, 0, 0], [1.0, 1, 1, 1]])
    assert torch.allclose(lab.token_ratio(logp, old), Lo.token_ratio(logp, old))
    assert torch.allclose(lab.sequence_ratio(logp, old, m), Lo.sequence_ratio(logp, old, m))
    rho = lab.token_ratio(logp, old)
    adv = torch.tensor([1.0, -1.0, 0.5])
    want, _ = Lo.clipped_surrogate(rho, adv, 0.2, 0.28)
    assert torch.allclose(lab.clipped_surrogate(rho, adv, 0.2, 0.28), want)
    lab.clipped_surrogate(rho, adv).sum().backward()
    assert logp.grad is not None and old.grad is None


def test_overlong_and_tis():
    L = torch.tensor([1.0, 6.0, 7.0, 8.0, 9.0])
    assert torch.allclose(lab.soft_overlong_penalty(L, 8, 2), Lo.soft_overlong_penalty(L, 8, 2))
    assert lab.soft_overlong_penalty(L, 8, 2).tolist() == [0.0, 0.0, -0.5, -1.0, -1.0]
    a, b = torch.log(torch.tensor([0.5, 0.2, 0.9])), torch.log(torch.tensor([0.1, 0.4, 0.9]))
    assert torch.allclose(lab.tis_weight(a, b, 2.0), torch.tensor([2.0, 0.5, 1.0]))
