import torch

from frontierlab.attention import accounting, bench, checks, ops, sliding  # noqa: F401  (registers kinds)
from frontierlab.labkit import load_target
from frontierlab.model import LM, baseline0, toy
from frontierlab.testing import cache_agreement, causal_check

lab = load_target(__file__)
V = 47


def lab_model(window=5, every=2):
    """toy model with the lab's local_global attention, weights copied from frontierlab's kind."""
    cfg = toy(vocab_size=V).with_(attention="local_global", extra={"window": window, "global_every": every})
    torch.manual_seed(0)
    ref = checks.sharpen(LM(cfg))
    m = LM(cfg)
    m.load_state_dict(ref.state_dict())
    for i, layer in enumerate(m.model.layers):
        sd = layer.self_attn.state_dict()
        layer.self_attn = lab.LabLocalGlobal(cfg, layer_idx=i)
        layer.self_attn.load_state_dict(sd)
    return m, ref


def test_window_mask_by_hand():
    m = lab.window_mask(torch.arange(5), torch.arange(5), 2)
    expect = torch.tensor([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]]).bool()
    assert torch.equal(m, expect)
    q, k = torch.arange(10, 13), torch.arange(3, 13)                 # decode: 3 new queries, 10 keys
    assert torch.equal(lab.window_mask(q, k, 4), ops.band_mask(q, k, 4))
    assert torch.equal(lab.window_mask(q, k, None), ops.band_mask(q, k, None))


def test_update_window_cache_keeps_the_window_and_returns_everything_needed():
    cache = {}
    k = torch.randn(1, 2, 6, 4)
    ka, _, pa = lab.update_window_cache(cache, k, k.clone(), torch.arange(6), 4)
    assert ka.shape[2] == 6 and torch.equal(pa, torch.arange(6))       # the prefix attends to all 6
    assert cache["k"].shape[2] == 4 and torch.equal(cache["pos"], torch.arange(2, 6))
    k2 = torch.randn(1, 2, 3, 4)
    ka, _, pa = lab.update_window_cache(cache, k2, k2.clone(), torch.arange(6, 9), 4)
    assert torch.equal(pa, torch.arange(2, 9)) and torch.equal(cache["pos"], torch.arange(5, 9))


def test_lab_attention_matches_frontierlab_and_passes_the_suite():
    with checks.exact_rmsnorm():
        m, ref = lab_model()
        idx = torch.randint(0, V, (2, 21))
        assert (m.double()(idx).logits - ref.double()(idx).logits).abs().max() < 1e-12
        assert causal_check(m, V, T=20, split=11) < 1e-12
        for chunk in (1, 3, 7):
            assert cache_agreement(m, V, T=26, prefix=6, chunk=chunk) < 1e-10, chunk


def test_layer_pattern_and_kv_bytes():
    assert lab.layer_pattern(12, 6) == sliding.layer_types(
        baseline0().with_(attention="local_global", extra={"global_every": 6}))
    assert lab.layer_pattern(4, 2) == ["local", "global", "local", "global"]
    lg = baseline0().with_(attention="local_global", extra={"global_every": 6, "window": 1024})
    for S in (100, 1024, 32768, 1 << 20):
        assert lab.kv_bytes(lab.layer_pattern(12, 6), S, 1024, 1024) == accounting.kv_bytes(lg, S)
    m, _ = lab_model()
    cache = bench.prefill(m.eval(), 13, chunk=4)
    per_layer = 2 * 2 * 32 * 4                                        # K and V, 2 KV heads, d 32, fp32
    assert cache.nbytes() - 8 * sum(layer["pos"].numel() for layer in cache.layers) == \
        lab.kv_bytes(lab.layer_pattern(4, 2), 13, per_layer, 5)
