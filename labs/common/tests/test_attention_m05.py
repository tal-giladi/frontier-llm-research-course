"""Correctness suite for the Module 5 attention kinds (gdn, kda, hybrid, dsa) and their accounting.

Before any comparison counts (plan section 5, Stage B), every kind passes: op-level float64 gradcheck
(input and every parameter), the causal check, cached-decode agreement in 1-, 3- and 7-token steps,
plus: recurrent vs chunked-parallel equivalence for the linear kinds; DSA equal to dense attention when
k >= context, its gather path equal to its mask path, and the documented gradient separation between
the indexer and the main model; cache bytes equal to the formula.
"""

import math

import pytest
import torch

from frontierlab.attention import ATTENTION
from frontierlab.attention import accounting, bench, checks, deltanet, dsa, hybrid  # noqa: F401  (registers kinds)
from frontierlab.model import LM, baseline0, toy
from frontierlab.testing import cache_agreement, causal_check, equivalence

V = 53
KINDS = {
    "kda": {"chunk": 4},
    "kda-recurrent": {"linear_mode": "recurrent"},
    "kda-noconv": {"chunk": 4, "conv_size": 0},
    "gdn": {"chunk": 4},
    "hybrid": {"full_every": 2, "chunk": 4},
    "hybrid-gdn-gated": {"full_every": 4, "linear_kind": "gdn", "full_kind": "gated", "chunk": 3},
    "hybrid-dsa": {"full_every": 2, "full_kind": "dsa", "index_topk": 5, "chunk": 4},
    "dsa": {"index_topk": 5},
    "dsa-gather": {"index_topk": 5, "dsa_path": "gather", "gather_block": 4},
    "dsa-dense": {"index_topk": 5, "dsa_mode": "dense"},
}


@pytest.fixture(autouse=True)
def _exact_rmsnorm():
    with checks.exact_rmsnorm():
        yield


def cfg_for(name, **kw):
    c = toy(vocab_size=V).with_(attention=name.split("-")[0], extra=dict(KINDS[name]))
    return c.with_(**kw) if kw else c


def model_for(name, seed=0, **kw):
    torch.manual_seed(seed)
    return checks.sharpen(LM(cfg_for(name, **kw)), seed=seed)


def tiny_cfg(name):
    c = cfg_for(name, hidden_size=8, num_attention_heads=2, num_key_value_heads=1, head_dim=4)
    ex = dict(c.extra)
    if name.startswith("dsa") or "dsa" in name:
        ex.update(index_heads=2, index_head_dim=4, index_topk=3)
    return c.with_(extra=ex)


def rand_inputs(B=2, H=3, T=23, dk=5, dv=4, seed=0, strong=False):
    g = torch.Generator().manual_seed(seed)
    f = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)  # noqa: E731
    q, k, v = f(B, H, T, dk), deltanet.l2norm(f(B, H, T, dk)), f(B, H, T, dv)
    gate = -torch.rand(B, H, T, dk, generator=g, dtype=torch.float64) * (6.0 if strong else 1.5)
    beta = torch.rand(B, H, T, generator=g, dtype=torch.float64)
    return q, k, v, gate, beta, f(B, H, dk, dv)


# --------------------------------------------------------------------------- the suite, every kind

@pytest.mark.parametrize("name", sorted(KINDS))
def test_registered_causal_and_cache(name):
    assert name.split("-")[0] in ATTENTION
    m = model_for(name)
    assert causal_check(m, V, T=20, split=11) < 1e-12
    for chunk in (1, 3, 7):
        assert cache_agreement(m, V, T=26, prefix=6, chunk=chunk) < 1e-10, chunk


@pytest.mark.parametrize("name", sorted(KINDS))
def test_gradcheck_op_level_float64(name):
    cfg = tiny_cfg(name)
    layers = range(cfg.num_hidden_layers) if name.startswith("hybrid") else [0]
    kinds_seen = set()
    for i in layers:
        if name.startswith("hybrid"):
            t = hybrid.hybrid_layer_types(cfg)[i]
            if t in kinds_seen:
                continue
            kinds_seen.add(t)
        assert checks.op_gradcheck(cfg, layer_idx=i), i


# --------------------------------------------------------------------------- linear kinds

@pytest.mark.parametrize("chunk", [1, 2, 4, 7, 16, 64])
def test_recurrent_equals_chunked_op_level(chunk):
    q, k, v, g, beta, s0 = rand_inputs()
    o1, S1 = deltanet.delta_rule_recurrent(q, k, v, g, beta, s0)
    o2, S2 = deltanet.delta_rule_chunked(q, k, v, g, beta, s0, chunk)
    equivalence(o2, o1, atol=1e-12, what=f"outputs, chunk {chunk}")
    equivalence(S2, S1, atol=1e-12, what=f"final state, chunk {chunk}")


def test_recurrent_equals_chunked_with_strong_decay():
    """Cumulative decays of exp(-50) inside a chunk: the pairwise form stays exact."""
    q, k, v, g, beta, s0 = rand_inputs(strong=True, T=40)
    o1, _ = deltanet.delta_rule_recurrent(q, k, v, g, beta, s0)
    o2, _ = deltanet.delta_rule_chunked(q, k, v, g, beta, s0, 16)
    equivalence(o2, o1, atol=1e-12, what="outputs with strong decay")


def test_recurrent_equals_chunked_model_level():
    m = model_for("hybrid").double().eval()
    idx = torch.randint(0, V, (2, 21), generator=torch.Generator().manual_seed(4))
    a = m(idx).logits
    assert deltanet.set_linear_mode(m, "recurrent") == 2
    b = m(idx).logits
    equivalence(a, b, atol=1e-11, what="chunked vs recurrent logits")


def test_chunked_gradients_equal_recurrent_gradients():
    q, k, v, g, beta, s0 = rand_inputs(T=11)
    grads = []
    for fn in (deltanet.delta_rule_recurrent, lambda *a: deltanet.delta_rule_chunked(*a, chunk=4)):
        ins = [t.clone().requires_grad_(True) for t in (q, k, v, g, beta, s0)]
        o, S = fn(*ins)
        (o.sin().sum() + S.cos().sum()).backward()
        grads.append([t.grad for t in ins])
    for ga, gb in zip(*grads):
        equivalence(ga, gb, atol=1e-11, what="gradients")


def test_delta_rule_overwrites_where_linear_attention_accumulates():
    """With orthonormal keys, alpha = 1 and beta = 1, writing key k twice keeps only the second value."""
    k = torch.eye(4, dtype=torch.float64)[[0, 1, 0]].view(1, 1, 3, 4)     # keys e0, e1, e0
    v = torch.tensor([[1.0, 0], [0, 1], [5, 5]], dtype=torch.float64).view(1, 1, 3, 2)
    g = torch.zeros(1, 1, 3, 4, dtype=torch.float64)
    beta = torch.ones(1, 1, 3, dtype=torch.float64)
    _, S = deltanet.delta_rule_recurrent(k, k, v, g, beta)
    assert torch.allclose(S[0, 0].T @ k[0, 0, 0], torch.tensor([5.0, 5.0], dtype=torch.float64))  # not 6, 5
    S_lin = torch.einsum("tk,tv->kv", k[0, 0], v[0, 0])                     # plain linear attention
    assert torch.allclose(S_lin.T @ k[0, 0, 0], torch.tensor([6.0, 5.0], dtype=torch.float64))


def test_scalar_gate_is_kda_with_equal_channels():
    q, k, v, g, beta, s0 = rand_inputs()
    g_scalar = g[..., :1].expand_as(g)
    o1, _ = deltanet.delta_rule_chunked(q, k, v, g_scalar, beta, s0, 5)
    o2, _ = deltanet.delta_rule_recurrent(q, k, v, g_scalar, beta, s0)
    equivalence(o1, o2, atol=1e-12)


@pytest.mark.parametrize("name", ["kda", "gdn", "hybrid", "hybrid-dsa", "dsa"])
def test_cache_bytes_match_formula_and_linear_state_is_constant(name):
    m = model_for(name).eval()
    for S in (5, 13, 23):
        cache = bench.prefill(m, S, chunk=4)
        assert cache.nbytes() == accounting.m05_cache_bytes(m.config, S, bytes_per=4, state_bytes=4,
                                                           include_pos=True), S
        for layer, kind in zip(cache.layers, accounting.m05_layer_kinds(m.config)):
            if kind == "linear":
                assert set(layer) == {"state", "conv"}
    for kind, layer in zip(accounting.m05_layer_kinds(m.config), cache.layers):
        if kind == "linear":
            assert layer["state"].shape == (1, 4, 32, 32) and layer["state"].dtype == torch.float32


def test_state_stays_fp32_under_bf16():
    cfg = cfg_for("kda")
    m = LM(cfg).to(torch.bfloat16).eval()
    cache = m.new_cache()
    with torch.no_grad():
        m(torch.randint(0, V, (1, 9)), cache=cache)
    assert cache.layers[0]["state"].dtype == torch.float32 and cache.layers[0]["conv"].dtype == torch.bfloat16


# --------------------------------------------------------------------------- hybrid layouts

def test_hybrid_layouts_of_qwen3_next_and_kimi_linear():
    qn = baseline0().with_(num_hidden_layers=48, attention="hybrid", extra={"full_every": 4})
    t = hybrid.hybrid_layer_types(qn)
    assert t[:4] == ["linear"] * 3 + ["full"] and t.count("full") == 12 and t.count("linear") == 36
    kimi = hybrid.from_full_layer_list([4, 8, 12, 16, 20, 24, 27], 27)
    assert kimi.count("full") == 7 and kimi.count("linear") == 20 and kimi[-3:] == ["linear", "linear", "full"]
    c = baseline0().with_(num_hidden_layers=27, attention="hybrid", extra={"layer_types": kimi})
    assert hybrid.hybrid_layer_types(c) == kimi


def test_hybrid_full_layers_are_the_inner_kind():
    m = LM(cfg_for("hybrid-gdn-gated"))
    kinds = [type(layer.self_attn.inner).__name__ for layer in m.model.layers]
    assert kinds == ["GatedDeltaNet"] * 3 + ["GatedAttention"]


# --------------------------------------------------------------------------- DSA

def _dsa_and_dense(topk, T=24, path="mask"):
    torch.manual_seed(0)
    base = checks.sharpen(LM(toy(vocab_size=V)))
    m = LM(toy(vocab_size=V).with_(attention="dsa", extra={"index_topk": topk, "dsa_path": path}))
    missing, unexpected = m.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(".idx_" in n for n in missing)
    return m.double().eval(), base.double().eval()


def test_dsa_with_k_at_least_context_is_dense_attention():
    m, base = _dsa_and_dense(topk=24)
    idx = torch.randint(0, V, (2, 24), generator=torch.Generator().manual_seed(2))
    equivalence(m(idx).logits, base(idx).logits, atol=1e-12, what="DSA k>=T vs dense")
    m2, _ = _dsa_and_dense(topk=6)
    assert (m2(idx).logits - base(idx).logits).abs().max() > 1e-3          # selection really changes things


def test_dsa_gather_path_equals_mask_path():
    m, _ = _dsa_and_dense(topk=6)
    idx = torch.randint(0, V, (2, 24), generator=torch.Generator().manual_seed(3))
    a = m(idx).logits
    dsa.set_dsa(m, path="gather")
    for layer in dsa.dsa_layers(m):
        layer.gather_block = 5
    equivalence(m(idx).logits, a, atol=1e-12, what="gather vs mask")


def test_topk_mask_by_hand():
    scores = torch.tensor([3.0, 1.0, 2.0, 9.0]).expand(1, 4, 4).clone()   # same scores for every query
    allowed = torch.ones(4, 4, dtype=torch.bool).tril()[None]
    sel, _ = dsa.topk_mask(scores, allowed, 2)
    expect = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]], dtype=torch.bool)
    assert torch.equal(sel[0], expect)


def test_indexer_kl_by_hand_and_zero_at_match():
    p = torch.tensor([[[0.5, 0.25, 0.25]]])
    support = torch.ones(1, 1, 3, dtype=torch.bool)
    assert dsa.indexer_kl(p, p.log(), support).abs() < 1e-7                # softmax(log p) = p
    I = torch.zeros(1, 1, 3)                                               # uniform indexer
    expect = 0.5 * math.log(0.5 * 3) + 2 * 0.25 * math.log(0.25 * 3)
    assert abs(dsa.indexer_kl(p, I, support).item() - expect) < 1e-6
    # restricted to a support of two keys: p renormalised to (2/3, 1/3), softmax over the two
    sub = torch.tensor([[[True, True, False]]])
    expect2 = (2 / 3) * math.log((2 / 3) / 0.5) + (1 / 3) * math.log((1 / 3) / 0.5)
    assert abs(dsa.indexer_kl(p, I, sub).item() - expect2) < 1e-6


@pytest.mark.parametrize("mode", ["dense", "sparse"])
def test_gradient_separation_between_indexer_and_main_model(mode):
    """L^I trains only the indexer; the LM loss trains only the main model (V3.2 section 2.1.1)."""
    torch.manual_seed(0)
    m = LM(toy(vocab_size=V).with_(attention="dsa", extra={"index_topk": 6, "dsa_mode": mode}))
    dsa.set_dsa(m, collect=True)
    idx = torch.randint(0, V, (2, 20))
    out = m(idx, labels=idx)
    li = dsa.indexer_loss(m)
    assert li is not None and li.item() > 0
    idx_params = {id(p) for p in dsa.indexer_parameters(m)}
    li.backward(retain_graph=True)
    for n, p in m.named_parameters():
        if id(p) in idx_params:
            assert p.grad is not None and p.grad.abs().sum() > 0, n
        else:
            assert p.grad is None or p.grad.abs().sum() == 0, n
    m.zero_grad(set_to_none=True)
    out.loss.backward()
    for n, p in m.named_parameters():
        if id(p) in idx_params:
            assert p.grad is None or p.grad.abs().sum() == 0, n


def test_selection_recall_is_one_when_k_covers_the_context():
    torch.manual_seed(0)
    m = LM(toy(vocab_size=V).with_(attention="dsa", extra={"index_topk": 64})).eval()
    rec = dsa.selection_recall(m, torch.randint(0, V, (1, 30)))
    assert all(abs(r[key] - 1.0) < 1e-6 for r in rec for key in ("indexer", "oracle", "window"))
    rec4 = dsa.selection_recall(m, torch.randint(0, V, (1, 30)), topk=4)
    assert all(0 < r["indexer"] <= r["oracle"] + 1e-9 and r["window"] <= r["oracle"] + 1e-9 for r in rec4)


# --------------------------------------------------------------------------- accounting

@pytest.mark.parametrize("name", ["kda", "gdn", "hybrid", "hybrid-dsa", "dsa", "hybrid-gdn-gated"])
def test_param_counts_exact(name):
    m = LM(cfg_for(name))
    assert accounting.param_counts(m.config)["total"] == sum(p.numel() for p in m.parameters())


def test_m05_flops_equal_module3_flops_for_old_kinds():
    for c in (baseline0(), baseline0().with_(attention="local_global", extra={"window": 256, "global_every": 6})):
        assert accounting.m05_flops_per_token(c, 1024) == accounting.flops_per_token(c, 1024)


def test_dsa_crossover_root():
    c = baseline0().with_(attention="dsa", extra={"index_topk": 2048})
    L = accounting.dsa_crossover_length(c)
    dense = accounting.m05_mix_flops(baseline0(), L)["full"]
    sparse = accounting.m05_mix_flops(c, L)
    assert abs(dense - sparse["dsa_indexer"] - sparse["dsa_attention"]) / dense < 1e-9
    assert L > 2048


def test_linear_crossover():
    c = baseline0().with_(attention="kda", extra={"chunk": 64})
    L = accounting.linear_crossover_length(c)
    assert abs(accounting.linear_mix_flops(c) - 4 * 12 * 64 * L / 2) < 1e-6


def test_decode_flops_linear_constant_in_context():
    c = baseline0().with_(attention="kda")
    assert accounting.m05_decode_flops_per_token(c, 1024) == accounting.m05_decode_flops_per_token(c, 1 << 20)


# --------------------------------------------------------------------------- main path (GPU) only

@pytest.mark.skipif(not torch.cuda.is_available(), reason="flash-linear-attention needs CUDA (Module 5 pilot)")
@pytest.mark.parametrize("gate", ["scalar", "channel"])
def test_fla_matches_reference(gate):
    """NOT RUN IN THIS BUILD. The pilot's first check: fla 0.5.2 chunk kernels vs our float32 reference."""
    pytest.importorskip("fla")
    torch.manual_seed(0)
    B, H, T, dk = 2, 4, 200, 64
    q = torch.randn(B, H, T, dk, device="cuda") * dk ** -0.5
    k = deltanet.l2norm(torch.randn(B, H, T, dk, device="cuda"))
    v = torch.randn(B, H, T, dk, device="cuda")
    g = -torch.rand(B, H, T, dk if gate == "channel" else 1, device="cuda").expand(B, H, T, dk) * 0.5
    beta = torch.rand(B, H, T, device="cuda")
    cfg = toy().with_(attention="kda" if gate == "channel" else "gdn", head_dim=dk)
    layer = ATTENTION[cfg.attention](cfg).cuda()
    ref, Sref = deltanet.delta_rule_chunked(q, k, v, g, beta, None, 16)
    out, S = layer._fla(q, k, v, g.contiguous(), beta, None)
    assert (out.float() - ref).abs().max() < 2e-3 and (S.float() - Sref).abs().max() < 2e-3


# --------------------------------------------------------------------------- 05.3 component benchmark, 05.4 compression

@pytest.mark.parametrize("decode", [False, True])
def test_blocked_dsa_components_equal_the_reference(decode):
    from frontierlab.attention import subq_bench as sb
    sh = sb.Shape(H=4, KV=2, d=8, HI=2, dI=4, k=7)
    x = sb.make_inputs(sh, 40, decode=decode, dtype=torch.float64)
    equivalence(sb.dsa_full(x, sh, block=6), sb.dsa_reference(x, sh), atol=1e-12, what="blocked DSA")


def test_component_model_counts():
    from frontierlab.attention import subq_bench as sb
    sh = sb.Shape(H=12, KV=4, d=64, HI=4, dI=32, k=2048)
    fb = sb.component_flops_bytes(sh, 131072, decode=True, b=2)
    assert fb["dense.sdpa"] == (4 * 12 * 64 * 131072, 2 * (2 * 4 * 131072 * 64 + 2 * 12 * 64))
    assert fb["dsa.attend"][0] == 4 * 12 * 64 * 2048
    pre = sb.component_flops_bytes(sh, 4096, decode=False, b=2)
    assert pre["dsa.attend"][0] == 4 * 12 * 64 * (2048 * 2049 / 2 + 2048 * 2048)


def test_compression_is_causal_and_overlapping():
    from frontierlab.attention import compressed
    g = torch.Generator().manual_seed(0)
    c, z = torch.randn(1, 16, 3, generator=g, dtype=torch.float64), torch.randn(1, 16, 3, generator=g, dtype=torch.float64)
    assert torch.allclose(compressed.compress_hca(c, torch.zeros_like(z), 4), c.view(1, 4, 4, 3).mean(2))
    out = compressed.compress_csa(c, z, c, z, 4)
    c2 = c.clone()
    c2[:, 12:] += 1.0                                                    # change the last block
    assert torch.equal(compressed.compress_csa(c2, z, c2, z, 4)[:, :3], out[:, :3])
    assert compressed.usable_entries(3, 4) == 1 and compressed.usable_entries(2, 4) == 0
    assert compressed.attended_entries(1 << 20, kind="csa", m=4, k=512, window=128) == 640
