import torch

from frontierlab.attention import accounting, bench, checks, deltanet, hybrid  # noqa: F401  (registers kinds)
from frontierlab.labkit import load_target
from frontierlab.model import LM, baseline0, toy
from frontierlab.testing import cache_agreement, causal_check, equivalence

lab = load_target(__file__)
V = 47


def rand(B=2, H=3, T=21, dk=5, dv=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)  # noqa: E731
    return (f(B, H, T, dk), deltanet.l2norm(f(B, H, T, dk)), f(B, H, T, dv),
            -torch.rand(B, H, T, dk, generator=g, dtype=torch.float64) * 2,
            torch.rand(B, H, T, generator=g, dtype=torch.float64), f(B, H, dk, dv))


def test_recurrent_step_matches_reference():
    q, k, v, g, beta, s0 = rand()
    o_ref, S_ref = deltanet.delta_rule_recurrent(q[:, :, :1], k[:, :, :1], v[:, :, :1], g[:, :, :1], beta[:, :, :1], s0)
    o, S = lab.recurrent_step(s0.clone(), q[:, :, 0], k[:, :, 0], v[:, :, 0], g[:, :, 0], beta[:, :, 0])
    equivalence(o, o_ref[:, :, 0], atol=1e-13)
    equivalence(S, S_ref, atol=1e-13)
    o_all, _ = lab.delta_recurrent(q, k, v, g, beta, s0)
    equivalence(o_all, deltanet.delta_rule_recurrent(q, k, v, g, beta, s0)[0], atol=1e-12)


def test_one_chunk_matches_reference():
    q, k, v, g, beta, s0 = rand(T=8)
    O, S = lab.chunk_forward(q, k, v, g, beta, s0)
    O_ref, S_ref = deltanet.delta_rule_chunked(q, k, v, g, beta, s0, chunk=8)
    equivalence(O, O_ref, atol=1e-12, what="chunk outputs")
    equivalence(S, S_ref, atol=1e-12, what="chunk final state")


def test_chunked_equals_recurrent_for_every_chunk_size():
    q, k, v, g, beta, s0 = rand(T=23)
    o_rec, S_rec = deltanet.delta_rule_recurrent(q, k, v, g, beta, s0)
    for chunk in (1, 3, 4, 16, 64):
        o, S = lab.delta_chunked(q, k, v, g, beta, s0, chunk)
        equivalence(o, o_rec, atol=1e-12, what=f"chunk {chunk}")
        equivalence(S, S_rec, atol=1e-12, what=f"state, chunk {chunk}")


def test_strong_decay_does_not_overflow():
    q, k, v, g, beta, s0 = rand(T=32)
    g = g * 20                                                        # cumulative decay down to exp(-600)
    o, _ = lab.delta_chunked(q, k, v, g, beta, s0, 16)
    assert torch.isfinite(o).all()
    equivalence(o, deltanet.delta_rule_recurrent(q, k, v, g, beta, s0)[0], atol=1e-12)


def lab_model(mode="chunked"):
    cfg = toy(vocab_size=V).with_(attention="kda", extra={"chunk": 4, "linear_mode": mode})
    torch.manual_seed(0)
    ref = checks.sharpen(LM(cfg))
    m = LM(cfg)
    m.load_state_dict(ref.state_dict())
    for layer in m.model.layers:
        sd = layer.self_attn.state_dict()
        layer.self_attn = lab.LabKDA(cfg)
        layer.self_attn.load_state_dict(sd)
    return m, ref


def test_lab_layer_matches_frontierlab_and_passes_the_suite():
    with checks.exact_rmsnorm():
        for mode in ("chunked", "recurrent"):
            m, ref = lab_model(mode)
            idx = torch.randint(0, V, (2, 19), generator=torch.Generator().manual_seed(1))
            assert (m.double()(idx).logits - ref.double()(idx).logits).abs().max() < 1e-11
            assert causal_check(m, V, T=20, split=11) < 1e-12
            for chunk in (1, 3, 7):
                assert cache_agreement(m, V, T=26, prefix=6, chunk=chunk) < 1e-10, (mode, chunk)


def test_hybrid_pattern_and_cache_bytes():
    assert lab.hybrid_pattern(4, 4) == ["linear", "linear", "linear", "full"]
    c = baseline0().with_(num_hidden_layers=48, attention="hybrid", extra={"full_every": 4})
    assert lab.hybrid_pattern(48, 4) == hybrid.hybrid_layer_types(c)
    cfg = baseline0().with_(attention="hybrid", extra={"full_every": 4})
    kv = 2 * 4 * 64 * 2                                               # K and V, 4 KV heads, d 64, BF16
    st = accounting.linear_state_bytes(cfg, bytes_per=2, state_bytes=4)
    for S in (1, 1024, 32768, 1 << 20):
        assert lab.cache_bytes(lab.hybrid_pattern(12, 4), S, kv, st) == accounting.m05_cache_bytes(cfg, S)
