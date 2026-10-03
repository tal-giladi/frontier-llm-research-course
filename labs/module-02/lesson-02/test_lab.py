import pytest
import torch
import torch.nn.functional as F

from frontierlab.labkit import load_target
from frontierlab.model import LM, toy
from frontierlab.perf.memory import saved_activation_bytes

lab = load_target(__file__)


def test_ce_and_grads_matches_autograd_in_float64():
    torch.manual_seed(0)
    N, C, V = 29, 8, 40
    h = torch.randn(N, C, dtype=torch.float64, requires_grad=True)
    w = torch.randn(V, C, dtype=torch.float64, requires_grad=True)
    t = torch.randint(0, V, (N,))
    ref = F.cross_entropy(h @ w.t(), t)
    gh, gw = torch.autograd.grad(ref, (h, w))
    for chunk in (1, 7, 29, 64):
        loss, dh, dw = lab.ce_and_grads(h.detach(), w.detach(), t, chunk)
        assert abs(float(loss) - ref.item()) < 1e-12
        assert dh.shape == h.shape and dw.shape == w.shape
        assert (dh - gh).abs().max() < 1e-12 and (dw - gw).abs().max() < 1e-12


def test_checkpointed_loss_same_value_and_gradients():
    torch.manual_seed(0)
    m = LM(toy(vocab_size=97)).double()
    x = torch.randint(0, 97, (2, 24))
    ref = m(x, labels=x).loss
    g_ref = torch.autograd.grad(ref, list(m.parameters()))
    out = lab.checkpointed_loss(m, x)
    g = torch.autograd.grad(out, list(m.parameters()))
    assert abs(out.item() - ref.item()) < 1e-6        # LM casts logits to fp32, so compare at fp32 level
    assert max((a - b).abs().max().item() for a, b in zip(g, g_ref)) < 1e-6


def test_checkpointing_saves_activation_memory():
    torch.manual_seed(0)
    m = LM(toy(vocab_size=512))
    x = torch.randint(0, 512, (4, 64))
    plain = saved_activation_bytes(m, lambda: m(x, labels=x).loss).total_bytes
    ckpt = saved_activation_bytes(m, lambda: lab.checkpointed_loss(m, x)).total_bytes
    assert ckpt < 0.5 * plain


def test_categorize():
    top = [("aten::mm", 5.0, 10, 0.5), ("aten::_scaled_dot_product_flash_attention_for_cpu", 1.0, 4, 0.1),
           ("aten::_log_softmax", 1.0, 1, 0.1), ("aten::_foreach_mul_", 0.5, 2, 0.05), ("aten::mul", 2.0, 30, 0.2)]
    s = lab.categorize(top)
    assert s == pytest.approx({"matmul": 0.5, "attention": 0.1, "loss": 0.1, "optimizer": 0.05, "other": 0.2})
    assert lab.categorize([])["other"] == 0.0
