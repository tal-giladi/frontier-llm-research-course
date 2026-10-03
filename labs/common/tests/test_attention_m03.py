"""Correctness suite for the Module 3 attention kinds (mla, sliding, local_global, sink, gated, gqa_partial).

Every kind must pass, before any comparison counts (plan section 5, Stage B):
float64 gradient check at op level (inputs AND parameters), the causal check, cached-decode agreement
with 1-token and chunked steps, plus kind-specific checks (MLA naive vs absorbed; window cache bounded;
sinks and gates reducing to Baseline-0 in their limits; cache bytes equal to the analytic formula).
"""

import pytest
import torch

from frontierlab.attention import ATTENTION
from frontierlab.attention import accounting, bench, checks, gated, headshape, mla, ops, probes, sliding  # noqa: F401
from frontierlab.flops import flops_per_token as b0_flops
from frontierlab.model import LM, ModelConfig, baseline0, param_counts, toy
from frontierlab.testing import cache_agreement, causal_check, equivalence, grad_check

V = 53
KINDS = {
    "mla": {"kv_lora_rank": 24, "qk_rope_head_dim": 8},
    "mla-qlora": {"kv_lora_rank": 24, "qk_rope_head_dim": 8, "q_lora_rank": 20},
    "mla-absorbed": {"kv_lora_rank": 24, "qk_rope_head_dim": 8, "mla_mode": "absorbed"},
    "sliding": {"window": 5},
    "local_global": {"window": 5, "global_every": 2},
    "local_global-sinks": {"window": 5, "global_every": 2, "sinks": True},
    "sink": {},
    "gated": {},
    "gated-headwise": {"gate": "headwise"},
    "gated-window": {"window": 6},
    "gqa_partial": {"rope_fraction": 0.5},
}


@pytest.fixture(autouse=True)
def _exact_rmsnorm():
    """Every check in this file runs with RMSNorm in the input dtype (see frontierlab.attention.checks)."""
    with checks.exact_rmsnorm():
        yield


def cfg_for(name, **kw):
    kind = name.split("-")[0]
    c = toy(vocab_size=V).with_(attention=kind, extra=dict(KINDS[name]))
    return c.with_(**kw) if kw else c


def model_for(name, seed=0, **kw):
    """A toy model of the kind with sharpened weights (O(1), peaked attention; see checks.sharpen)."""
    torch.manual_seed(seed)
    return checks.sharpen(LM(cfg_for(name, **kw)), seed=seed)


def tiny_cfg(name):
    c = cfg_for(name, hidden_size=8, num_attention_heads=2, num_key_value_heads=1, head_dim=4)
    ex = dict(c.extra)
    if "kv_lora_rank" in ex:
        ex.update(kv_lora_rank=6, qk_rope_head_dim=2, **({"q_lora_rank": 5} if "q_lora_rank" in ex else {}))
    if "window" in ex:
        ex["window"] = 3
    return c.with_(extra=ex)


@pytest.mark.parametrize("name", sorted(KINDS))
def test_registered_and_causal_and_cache(name):
    assert name.split("-")[0] in ATTENTION
    m = model_for(name)
    assert causal_check(m, V, T=20, split=11) < 1e-12
    for chunk in (1, 3, 7):            # chunks that cross window boundaries
        assert cache_agreement(m, V, T=26, prefix=6, chunk=chunk) < 1e-10, chunk


@pytest.mark.parametrize("name", sorted(KINDS))
def test_gradcheck_op_level_float64(name):
    """gradcheck one attention module w.r.t. its input AND every parameter, in float64."""
    assert checks.op_gradcheck(tiny_cfg(name))


def test_gradcheck_catches_a_wrong_backward():
    """The op-level check must fail for a function whose backward is wrong."""
    class Bad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x * x

        @staticmethod
        def backward(ctx, g):
            return g                                   # should be 2x * g
    with pytest.raises(Exception):
        grad_check(Bad.apply, torch.randn(4))


def test_attention_suite_on_mla():
    m = model_for("mla")
    res = checks.attention_suite(m, tiny_cfg("mla"), V, log=lambda s: None)
    assert res["naive_vs_absorbed"] < 1e-10 and res["cache_chunk5"] < 1e-9


def test_mla_naive_equals_absorbed_full_and_cached():
    m = model_for("mla").double()
    idx = torch.randint(0, V, (2, 18), generator=torch.Generator().manual_seed(3))
    naive = m(idx).logits
    mla.set_mla_mode(m, "absorbed")
    absorbed = m(idx).logits
    assert equivalence(naive, absorbed, atol=1e-11, what="naive vs absorbed MLA logits") < 1e-11
    assert cache_agreement(m, V, T=20, prefix=5, chunk=1) < 1e-10      # absorbed decode == full forward


def test_mla_caches_latent_only_and_bytes_match_formula():
    m = model_for("mla")
    cache = bench.prefill(m.eval(), 17)
    names = {k for layer in cache.layers for k in layer}
    assert names == {"c_kv", "k_rope", "pos"}
    assert cache.nbytes() == accounting.cache_bytes(m.config, 17, bytes_per=4)
    assert cache.layers[0]["c_kv"].shape == (1, 17, 24) and cache.layers[0]["k_rope"].shape == (1, 1, 17, 8)


@pytest.mark.parametrize("name", ["sliding", "local_global", "local_global-sinks", "gated-window"])
def test_window_cache_is_bounded_and_matches_formula(name):
    m = model_for(name).eval()
    for S in (4, 5, 23):
        cache = bench.prefill(m, S, chunk=4)
        assert cache.nbytes() == accounting.cache_bytes(m.config, S, bytes_per=4), S
        w = m.config.extra["window"]
        for layer, kind in zip(cache.layers, accounting.layer_kinds(m.config)):
            assert layer["k"].shape[2] == (min(S, w) if kind == "local" else S)


def test_local_global_layer_pattern():
    c = baseline0().with_(attention="local_global", extra={"global_every": 6, "window": 1024})
    assert sliding.layer_types(c) == ["local"] * 5 + ["global"] + ["local"] * 5 + ["global"]   # Gemma 3: 5:1
    c2 = c.with_(extra={"global_every": 2})
    assert sliding.layer_types(c2)[:4] == ["local", "global", "local", "global"]               # gpt-oss: alternating


def test_window_at_least_T_equals_full_attention():
    torch.manual_seed(0)
    base = LM(toy(vocab_size=V)).double()
    wide = LM(toy(vocab_size=V).with_(attention="sliding", extra={"window": 64})).double()
    wide.load_state_dict(base.state_dict())
    idx = torch.randint(0, V, (2, 30))
    assert equivalence(base(idx).logits, wide(idx).logits, atol=1e-12) < 1e-12


def test_sink_to_minus_infinity_is_baseline_and_sink_mass_is_positive():
    torch.manual_seed(0)
    base = LM(toy(vocab_size=V)).double()
    s = LM(cfg_for("sink")).double()
    s.load_state_dict({**base.state_dict(), **{k: v for k, v in s.state_dict().items() if "sinks" in k}})
    idx = torch.randint(0, V, (2, 15))
    for layer in s.model.layers:
        layer.self_attn.sinks.data.fill_(-1e4)
    assert equivalence(base(idx).logits, s(idx).logits, atol=1e-12) < 1e-12
    for layer in s.model.layers:
        layer.self_attn.sinks.data.fill_(2.0)
    rep = probes.report(s, idx, skip=2)
    assert rep["sink_mass"] > 0.05 and abs(probes.report(base, idx, skip=2)["sink_mass"]) < 1e-6


def test_sink_softmax_by_hand():
    logits = torch.tensor([[[[0.0, 0.0]]]])                  # one head, one query, two keys
    p = ops.sink_softmax(logits, torch.tensor([0.0]))
    assert torch.allclose(p, torch.tensor([[[[1 / 3, 1 / 3]]]]))


def test_gate_param_counts_and_open_gate_limit():
    c = toy(vocab_size=V)
    pe = accounting.param_counts(c.with_(attention="gated"))["non_embedding"]
    ph = accounting.param_counts(c.with_(attention="gated", extra={"gate": "headwise"}))["non_embedding"]
    p0 = param_counts(c)["non_embedding"]
    L, C, H, d = c.num_hidden_layers, c.hidden_size, c.num_attention_heads, c.head_dim
    assert pe - p0 == L * C * H * d and ph - p0 == L * C * H
    torch.manual_seed(0)
    base = LM(c).double()
    g = LM(c.with_(attention="gated", extra={"gate": "headwise"})).double()
    sd = g.state_dict()
    sd.update(base.state_dict())
    g.load_state_dict(sd)
    for layer in g.model.layers:                 # W_g = 0: every gate is sigmoid(0) = 0.5
        layer.self_attn.gate_proj.weight.data.zero_()
    idx = torch.randint(0, V, (1, 9))
    # a constant gate of 0.5 halves the attention output: the same as Baseline-0 with o_proj halved
    half = LM(c).double()
    sd = base.state_dict()
    half.load_state_dict({k: (0.5 * v if "o_proj" in k else v) for k, v in sd.items()})
    assert equivalence(half(idx).logits, g(idx).logits, atol=1e-12) < 1e-12


def test_partial_rope_rotates_only_a_prefix():
    c = toy(vocab_size=V).with_(attention="gqa_partial", extra={"rope_fraction": 0.25})
    m = LM(c)
    assert m.model.layers[0].self_attn.rot == 8 and headshape.rotary_width(256, 0.25) == 64


def test_accounting_matches_modules_and_baseline_flops():
    for name in KINDS:
        c = cfg_for(name)
        assert accounting.param_counts(c)["total"] == sum(p.numel() for p in LM(c).parameters()), name
    b0 = baseline0()
    assert accounting.flops_per_token(b0, 1024) == b0_flops(b0, 1024)
    assert accounting.param_counts(b0)["non_embedding"] == param_counts(b0)["non_embedding"]


def test_mla_param_count_by_hand_baseline0_shape():
    c = baseline0().with_(attention="mla", extra={"kv_lora_rank": 256, "qk_rope_head_dim": 32})
    attn = accounting.param_counts(c)["attention_per_layer"][0]
    C, H = 768, 12
    assert attn == C * H * 96 + C * 288 + 256 + 256 * H * 128 + H * 64 * C == 2_089_216


def test_kv_bytes_by_hand():
    b0 = baseline0()
    assert accounting.kv_bytes(b0, 32768) / 32768 == 12_288                    # lesson 01.1
    m = b0.with_(attention="mla", extra={"kv_lora_rank": 256, "qk_rope_head_dim": 32})
    assert accounting.kv_bytes(m, 32768) / 32768 == 12 * 288 * 2                  # 6,912
    lg = b0.with_(attention="local_global", extra={"global_every": 6, "window": 1024})
    assert accounting.kv_bytes(lg, 32768) == (2 * 32768 + 10 * 1024) * 512 * 2


def test_match_intermediate_and_equal_flops():
    b0 = baseline0()
    m = b0.with_(attention="mla", extra={"kv_lora_rank": 256, "qk_rope_head_dim": 32})
    target = param_counts(b0)["non_embedding"]
    m2, rel = accounting.match_intermediate(m, target)
    assert abs(rel) < 0.001 and m2.intermediate_size < b0.intermediate_size
    steps = accounting.equal_flops_steps(b0, m, 9500, 1024)
    assert steps < 9500
    assert abs(steps * accounting.flops_per_token(m, 1024) / (9500 * accounting.flops_per_token(b0, 1024)) - 1) < 1e-3


def test_decode_flops_absorbed_vs_naive_mla():
    m = baseline0().with_(attention="mla", extra={"kv_lora_rank": 256, "qk_rope_head_dim": 32})
    S = 32768
    naive, absorbed = accounting.decode_flops_per_token(m, S, "naive"), accounting.decode_flops_per_token(m, S, "absorbed")
    assert absorbed < naive                               # no re-expansion of the cache at every step


def test_decode_benchmark_runs_and_restores_cache():
    torch.manual_seed(0)
    models = {"gqa": LM(toy(vocab_size=V)).eval(), "mla": LM(cfg_for("mla")).eval()}
    rows = bench.decode_benchmark(models, [16], warmup=1, rounds=3, chunk=8, log=lambda s: None)
    assert [r["arm"] for r in rows] == ["gqa", "mla"]
    assert rows[0]["cache_bytes"] == accounting.cache_bytes(models["gqa"].config, 16, 4)
    assert rows[1]["kv_bytes"] == accounting.kv_bytes(models["mla"].config, 16, 4)
    assert "speedup_vs_first" in rows[1]


def test_probes_rows_and_uniform_reference():
    torch.manual_seed(0)
    m = LM(toy(vocab_size=V))
    idx = torch.randint(0, V, (2, 20))
    cap = probes.capture(m, idx)
    p = cap["probs"][0]
    assert p.shape == (2, 4, 20, 20) and torch.allclose(p.sum(-1), torch.ones(2, 4, 20))
    rep = probes.report(m, idx, skip=4)
    assert 0 < rep["first_token_mass"] < 1 and rep["massive_ratio"] >= 1
    assert abs(probes._uniform_reference(4, 0) - (1 + 1 / 2 + 1 / 3 + 1 / 4) / 4) < 1e-12
    mm = model_for("mla")
    assert probes.capture(mm, idx)["probs"][1].shape == (2, 4, 20, 20)
