import math

import torch

from frontierlab.labkit import load_target
from frontierlab.pipeline import distill as DI

lab = load_target(__file__)


def _logits(seed, shape=(2, 3, 5)):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float64)


def test_divergences_by_hand_and_against_reference():
    s = torch.log(torch.tensor([[[0.5, 0.5]]], dtype=torch.float64))
    t = torch.log(torch.tensor([[[0.9, 0.1]]], dtype=torch.float64))
    assert abs(float(lab.reverse_kl(s, t)) - (0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1))) < 1e-12
    assert abs(float(lab.forward_kl(s, t)) - (0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5))) < 1e-12
    a, b = _logits(0), _logits(1)
    assert torch.allclose(lab.reverse_kl(a, b), DI.exact_reverse_kl(a, b), atol=1e-12)
    assert torch.allclose(lab.forward_kl(a, b), DI.exact_forward_kl(a, b), atol=1e-12)
    assert lab.reverse_kl(a, b).shape == (2, 3) and float(lab.reverse_kl(a, a).abs().max()) < 1e-12


def test_teacher_gets_no_gradient():
    a, b = _logits(2).requires_grad_(True), _logits(3).requires_grad_(True)
    lab.reverse_kl(a, b).sum().backward()
    assert b.grad is None or float(b.grad.abs().max()) == 0.0


def test_sampled_loss_matches_reference_and_is_unbiased():
    # value and gradient equal the course's opd_loss on a batch
    g = torch.Generator().manual_seed(4)
    logp = (-torch.rand(3, 4, generator=g, dtype=torch.float64) * 3).requires_grad_(True)
    tl = -torch.rand(3, 4, generator=g, dtype=torch.float64) * 3
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.float64)
    mine = lab.sampled_rkl_loss(logp, tl, mask)
    ref, _ = DI.opd_loss(logp, logp.detach(), tl, mask)
    assert abs(float(mine.detach()) - float(ref.detach())) < 1e-12
    g1, = torch.autograd.grad(mine, logp)
    g2, = torch.autograd.grad(DI.opd_loss(logp, logp.detach(), tl, mask)[0], logp)
    assert torch.allclose(g1, g2, atol=1e-12)
    # expectation over the sampled token equals the gradient of the exact reverse KL (one position, V = 5)
    s = _logits(5, (5,)).requires_grad_(True)
    t = _logits(6, (5,))
    ls, lt = torch.log_softmax(s, -1), torch.log_softmax(t, -1)
    exp = torch.zeros(5, dtype=torch.float64)
    for y in range(5):
        loss = lab.sampled_rkl_loss(ls[y].reshape(1, 1), lt[y].reshape(1, 1), torch.ones(1, 1, dtype=torch.float64))
        gy, = torch.autograd.grad(loss, s, retain_graph=True)
        exp += ls[y].exp().detach() * gy
    true, = torch.autograd.grad(lab.reverse_kl(s[None, None], t[None, None]).sum(), s)
    assert torch.allclose(exp, true, atol=1e-12)


def test_arm_flops():
    f = lab.arm_flops(300_000, 1_200_000, student_sampled=1000, student_trained=1000, teacher_scored=1000)
    assert f["student_sample"] == 6e8 and f["student_train"] == 1.8e9 and f["teacher_score"] == 2.4e9
    assert f["teacher_sample"] == 0 and f["total"] == 4.8e9 and abs(f["teacher_share"] - 0.5) < 1e-12
    off = lab.arm_flops(300_000, 1_200_000, student_trained=1000, teacher_generated=1000)
    assert abs(off["teacher_share"] - 2.4e9 / 4.2e9) < 1e-12
    assert lab.arm_flops(1, 1)["teacher_share"] == 0.0
