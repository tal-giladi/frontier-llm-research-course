import torch

from frontierlab.model import LM, param_counts, toy
from frontierlab.testing import cache_agreement, causal_check, run_suite


def test_param_count_matches_module():
    cfg = toy()
    m = LM(cfg)
    n = sum(p.numel() for p in m.parameters())          # tied head counted once by .parameters()
    assert n == param_counts(cfg)["total"]


def test_causal_and_cache_suite():
    torch.manual_seed(0)
    cfg = toy(vocab_size=101)
    res = run_suite(LM(cfg), cfg.vocab_size)
    assert all(v < 1e-9 for v in res.values()), res


def test_cache_without_qk_norm_and_mha():
    torch.manual_seed(0)
    cfg = toy(vocab_size=50).with_(qk_norm=False, num_key_value_heads=4)
    m = LM(cfg)
    assert causal_check(m, 50) < 1e-9
    assert cache_agreement(m, 50, chunk=3) < 1e-9


def test_causal_check_catches_a_leak():
    """A deliberately non-causal model must fail the causal check."""
    import pytest
    from frontierlab.attention import gqa

    class Leaky(gqa.GQAttention):
        def forward(self, x, positions, cache=None):
            return super().forward(x, positions, cache) + x.mean(dim=1, keepdim=True)

    cfg = toy(vocab_size=40)
    m = LM(cfg)
    for layer in m.model.layers:
        sd = layer.self_attn.state_dict()
        layer.self_attn = Leaky(cfg)
        layer.self_attn.load_state_dict(sd)
    with pytest.raises(AssertionError):
        causal_check(m, 40)


def test_generate_greedy_matches_argmax_of_full_forward():
    torch.manual_seed(0)
    cfg = toy(vocab_size=37)
    m = LM(cfg).eval()
    idx = torch.randint(0, 37, (1, 5))
    out = m.generate(idx, 6)
    for t in range(5, 11):
        assert out[0, t] == m(out[:, :t]).logits[0, -1].argmax()
