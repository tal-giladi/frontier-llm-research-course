import torch

from frontierlab.labkit import load_target
from frontierlab.model import LM, toy
from frontierlab.testing import cache_agreement, causal_check

lab = load_target(__file__)


def lab_model(vocab=61, **kw):
    """Baseline-0 at toy size with the lab's attention swapped in (same weights)."""
    cfg = toy(vocab_size=vocab).with_(**kw)
    m = LM(cfg)
    for layer in m.model.layers:
        sd = layer.self_attn.state_dict()
        layer.self_attn = lab.LabGQAttention(cfg)
        layer.self_attn.load_state_dict(sd)
    return m


def test_count_params():
    for kw in ({}, {"qk_norm": False}, {"tie_word_embeddings": False}, {"num_key_value_heads": 4}):
        cfg = toy(vocab_size=500).with_(**kw)
        assert lab.count_params(cfg) == sum(p.numel() for p in LM(cfg).parameters()), kw


def test_attend_matches_full_attention_when_square():
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 4, 6, 8), torch.randn(1, 2, 6, 8), torch.randn(1, 2, 6, 8)
    pos = torch.arange(6)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    assert torch.allclose(lab.attend(q, k, v, pos, pos), ref, atol=1e-6)


def test_causal_and_cache():
    torch.manual_seed(0)
    m = lab_model()
    assert causal_check(m, 61) < 1e-9
    assert cache_agreement(m, 61, chunk=1) < 1e-9
    assert cache_agreement(m, 61, chunk=4) < 1e-9
