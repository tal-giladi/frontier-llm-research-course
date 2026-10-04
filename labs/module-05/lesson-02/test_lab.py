import math

import torch

from frontierlab.attention import checks, dsa
from frontierlab.labkit import load_target
from frontierlab.model import LM, toy
from frontierlab.testing import cache_agreement, causal_check, equivalence

lab = load_target(__file__)
V = 47


def test_index_scores_by_hand_and_against_reference():
    iq = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])                  # B=1, H_I=2, T=1, d_I=2
    ik = torch.tensor([[[2.0, -1.0], [-1.0, 3.0]]])                    # S=2
    w = torch.tensor([[[0.5, 2.0]]])                                   # (B, T, H_I)
    # key 0: 0.5*ReLU(2) + 2*ReLU(-1) = 1.0 ; key 1: 0.5*ReLU(-1) + 2*ReLU(3) = 6.0
    assert torch.allclose(lab.index_scores(iq, ik, w), torch.tensor([[[1.0, 6.0]]]))
    g = torch.Generator().manual_seed(0)
    iq, ik, w = torch.randn(2, 3, 5, 4, generator=g), torch.randn(2, 7, 4, generator=g), torch.randn(2, 5, 3, generator=g)
    assert torch.allclose(lab.index_scores(iq, ik, w), dsa.index_scores(iq, ik, w), atol=1e-6)


def test_select_topk():
    scores = torch.tensor([3.0, 1.0, 2.0, 9.0]).expand(1, 4, 4).clone()
    allowed = torch.ones(4, 4, dtype=torch.bool).tril()[None]
    expect = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]], dtype=torch.bool)
    assert torch.equal(lab.select_topk(scores, allowed, 2)[0], expect)
    assert torch.equal(lab.select_topk(scores, allowed, 10)[0], allowed[0])          # k >= S: dense causal


def test_target_and_kl():
    probs = torch.tensor([[[[0.5, 0.5, 0.0]], [[0.0, 0.5, 0.5]]]])     # B=1, H=2, T=1, S=3
    p = lab.indexer_target(probs)
    assert torch.allclose(p, torch.tensor([[[0.25, 0.5, 0.25]]]))
    full = torch.ones(1, 1, 3, dtype=torch.bool)
    assert lab.indexer_kl(p, p.log(), full).abs() < 1e-6
    I = torch.zeros(1, 1, 3)
    expect = 0.25 * math.log(0.75) + 0.5 * math.log(1.5) + 0.25 * math.log(0.75)
    assert abs(lab.indexer_kl(p, I, full).item() - expect) < 1e-6
    sub = torch.tensor([[[False, True, True]]])
    assert torch.allclose(lab.indexer_kl(p, I, sub), dsa.indexer_kl(p, I, sub), atol=1e-6)


def test_trainable_by_stage():
    names = ["model.layers.0.self_attn.idx_q.weight", "model.layers.0.self_attn.q_proj.weight", "lm_head.weight",
             "model.layers.3.self_attn.idx_w.weight", "model.norm.weight"]
    assert [lab.trainable(n, "warmup") for n in names] == [True, False, False, True, False]
    assert all(lab.trainable(n, "sparse") for n in names)


def lab_model(topk):
    cfg = toy(vocab_size=V).with_(attention="dsa", extra={"index_topk": topk})
    torch.manual_seed(0)
    ref = checks.sharpen(LM(cfg))
    m = LM(cfg)
    m.load_state_dict(ref.state_dict())
    for layer in m.model.layers:
        sd = layer.self_attn.state_dict()
        layer.self_attn = lab.LabDSA(cfg)
        layer.self_attn.load_state_dict(sd)
    return m, ref


def test_lab_dsa_matches_frontierlab_and_passes_the_suite():
    with checks.exact_rmsnorm():
        m, ref = lab_model(topk=5)
        idx = torch.randint(0, V, (2, 21), generator=torch.Generator().manual_seed(1))
        equivalence(m.double()(idx).logits, ref.double()(idx).logits, atol=1e-11)
        assert causal_check(m, V, T=20, split=11) < 1e-12
        for chunk in (1, 3, 7):
            assert cache_agreement(m, V, T=26, prefix=6, chunk=chunk) < 1e-10, chunk


def test_lab_indexer_loss_trains_only_the_indexer():
    m, _ = lab_model(topk=5)
    dsa.set_dsa(m, collect=True)
    idx = torch.randint(0, V, (2, 18))
    m(idx)
    li = sum(layer.self_attn.indexer_loss for layer in m.model.layers)
    li.backward()
    for n, p in m.named_parameters():
        has = p.grad is not None and p.grad.abs().sum() > 0
        assert has == (".idx_" in n), n
