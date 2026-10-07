import torch

from frontierlab.interp import hooks as HK
from frontierlab.interp import sae as S
from frontierlab.labkit import load_target
from frontierlab.model import LM, toy

lab = load_target(__file__)


def test_topk_by_hand_and_against_course():
    pre = torch.tensor([[0.5, -1.0, 2.0, 0.1], [-0.3, -0.2, -0.1, -0.4]])
    z = lab.topk_code(pre, 2)
    assert torch.equal(z, torch.tensor([[0.5, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0]]))   # negatives -> 0 by the ReLU
    g = torch.Generator().manual_seed(0)
    x = torch.randn(5, 7, 64, generator=g)
    sae = S.SAE(64, 64, "topk", k=6)
    assert torch.equal(lab.topk_code(x, 6), sae.code(x))
    assert int((lab.topk_code(x, 6) > 0).sum(-1).max()) <= 6


def test_jumprelu_forward_and_gradients_match_course():
    g = torch.Generator().manual_seed(1)
    pre = torch.randn(3, 4, 16, generator=g, dtype=torch.float64)
    theta = torch.rand(16, generator=g, dtype=torch.float64) * 0.5 + 0.1
    eps = 0.3                                            # wide enough that the rectangle catches some entries
    p1, t1 = pre.clone().requires_grad_(), theta.clone().requires_grad_()
    p2, t2 = pre.clone().requires_grad_(), theta.clone().requires_grad_()
    w = torch.randn(3, 4, 16, generator=g, dtype=torch.float64)
    (lab.JumpReLU.apply(p1, t1, eps) * w).sum().backward()
    (S.jumprelu(p2, t2, eps) * w).sum().backward()
    assert torch.allclose(lab.JumpReLU.apply(pre, theta, eps), S.jumprelu(pre, theta, eps))
    assert torch.allclose(p1.grad, p2.grad) and torch.allclose(t1.grad, t2.grad)
    assert t1.grad.abs().sum() > 0                      # the estimator really moves the threshold


def test_fvu():
    x = torch.tensor([[1.0, 0.0], [3.0, 2.0]])        # mean (2, 1); total variance 4
    assert abs(lab.fvu(x, x) - 0.0) < 1e-12
    assert abs(lab.fvu(x, x.mean(0).expand_as(x)) - 1.0) < 1e-12
    assert abs(lab.fvu(x, x + 1.0) - 1.0) < 1e-12       # error 4 / variance 4


def test_splice_check_on_a_model():
    torch.manual_seed(0)
    m = LM(toy(vocab_size=50).with_(num_hidden_layers=2)).double().eval()
    x = torch.randint(0, 50, (2, 12))
    sae = S.SAE(128, 256, "topk", k=8).double()
    clean = HK.run_with(m, x)
    with_err = HK.run_with(m, x, {"resid_post.0": lab.splice_edit(sae, keep_error=True)})
    spliced = HK.run_with(m, x, {"resid_post.0": lab.splice_edit(sae)})
    assert (with_err - clean).abs().max() < 1e-10
    assert (spliced - clean).abs().max() > 1e-3          # an untrained SAE changes the logits
    assert HK.n_hooks(m) == 0                            # no hook leaked
