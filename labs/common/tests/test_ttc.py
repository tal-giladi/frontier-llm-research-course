"""Tests for frontierlab.ttc (Module 15). No downloads; under a minute on a laptop."""

import itertools
import math

import numpy as np
import pytest
import torch

from frontierlab.model import LM, ModelConfig
from frontierlab.ttc import budget as BU
from frontierlab.ttc import kvquant as KQ
from frontierlab.ttc import select as SE
from frontierlab.ttc import serving as SV
from frontierlab.ttc import speculative as SP
from frontierlab.ttc import world as W


# --------------------------------------------------------------------------------------------- selection

def test_majority_vote_ties_and_none():
    assert SE.majority_vote(["3", "5", "5", "3"]) == "3"           # tie -> first occurrence
    assert SE.majority_vote([None, None, "7"]) == "7"              # None never wins against an answer
    assert SE.majority_vote([None, None]) is None
    assert SE.weighted_vote(["1", "2", "2"], [0.9, 0.4, 0.4]) == "1"
    assert SE.weighted_vote(["1", "2", "2"], [0.7, 0.4, 0.4]) == "2"
    assert SE.best_of_n(["a", "b", "c"], [0.1, 0.9, 0.9]) == "b"


def test_subset_success_full_pool_and_bounded_by_oracle():
    rng = np.random.default_rng(0)
    gold = ["1"] * 30
    answers = [[("1" if rng.random() < 0.35 else str(rng.integers(2, 4))) for _ in range(12)] for _ in gold]
    correct = np.array([[a == "1" for a in row] for row in answers])
    full = SE.subset_success(answers, gold, 12, "majority")
    assert np.allclose(full, [float(SE.majority_vote(r) == "1") for r in answers])
    for N in (1, 3, 6):
        sel = SE.subset_success(answers, gold, N, "majority", resamples=300, seed=1).mean()
        assert sel <= SE.oracle_pass_at_k(correct, N).mean() + 1e-9
    # N = 1: every procedure is a random draw, success = pass@1 = fraction correct
    one = SE.subset_success(answers, gold, 1, "majority", resamples=4000, seed=2)
    assert abs(one.mean() - correct.mean()) < 0.02


def test_oracle_pass_at_k_brute_force():
    row = np.array([[True, False, False, True, False]])
    for k in (1, 2, 3):
        brute = np.mean([row[0, list(s)].any() for s in itertools.combinations(range(5), k)])
        assert math.isclose(SE.oracle_pass_at_k(row, k)[0], brute)


def test_spend_and_matched_n():
    s = BU.sampling_spend(1000, prompt_tokens=10, gen_tokens=20, n=4, verifier_params=500, scored_tokens=30)
    assert s.prefill_tokens == 10 and s.decode_tokens == 80 and s.verifier_tokens == 120
    assert s.flops == 2 * 1000 * 90 + 2 * 500 * 120
    assert math.isclose(s.policy_token_equivalents, 90 + 60)
    assert BU.sampling_spend(1000, 10, 20, 4, shared_prefix=False).prefill_tokens == 40
    assert BU.matched_n(100, 20) == 5 and BU.matched_n(100, 20, 5) == 4


def test_latency_model_parallel_vs_sequential():
    lm = BU.LatencyModel({1: 1e-3, 16: 1.5e-3, 64: 4e-3}, prefill_per_token=1e-5, verifier={1: 2e-3, 64: 5e-3})
    # 8 chains of 20 tokens in a batch are much faster than one chain of 160 tokens
    assert lm.parallel(10, 20, 8) < lm.parallel(10, 160, 1)
    assert math.isclose(lm.decode(10, 1), 1e-2)
    assert lm.decode(1, 128) == pytest.approx(8e-3)                 # beyond the table: linear in batch


# --------------------------------------------------------------------------------------------- world

def test_world_trace_and_budget_forcing_format():
    p = W.SumProblem(4783, 3659)
    assert p.trace == "a12b14c14d08" and p.target() == "a12b14c14d08#8442"
    assert p.with_mode("h").trace.startswith("a3+9+0=12b8+5+1=14")
    assert p.with_mode("n").target() == "#8442"
    assert p.target(4) == "a12b#8442"
    assert W.parse("a12#84x") == ("a12", None) and W.parse("a12#8442") == ("a12", "8442")
    held = W.make_problems(50, 0, held=True)
    assert all(W.is_heldout(q.a, q.b) for q in held)
    assert not any(W.is_heldout(q.a, q.b) for q in W.make_problems(50, 0, held=False))


def test_sample_respects_budget_and_counts_tokens():
    from frontierlab.posttrain.sft import policy_config
    torch.manual_seed(0)
    m = LM(policy_config()).eval()
    probs = W.make_problems(6, 3, held=True)
    g = torch.Generator().manual_seed(0)
    for B in (0, 5):
        rows = W.sample(m, probs, n=3, think_budget=B, temperature=1.0, generator=g)
        for r in sum(rows, []):
            assert r["think_tokens"] <= B
            think, _ = W.parse(r["text"])
            assert len(think) <= r["think_tokens"]            # decode() drops special ids a random model may emit


def test_auc():
    assert W.auc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert W.auc([0.5, 0.5], [1, 0]) == 0.5


# --------------------------------------------------------------------------------------------- speculative sampling

def test_accept_reject_one_position_is_exact():
    """Emitted-token distribution of one round with gamma = 1 equals p, for random p and q (Monte Carlo)."""
    g = torch.Generator().manual_seed(0)
    V, B = 5, 200_000
    p = torch.softmax(torch.randn(V, generator=g, dtype=torch.float64) * 1.5, -1)
    q = torch.softmax(torch.randn(V, generator=g, dtype=torch.float64) * 1.5, -1)
    d = torch.multinomial(q.expand(B, V), 1, replacement=True, generator=g)
    u = torch.rand(B, 1, generator=g, dtype=torch.float64)
    pp = torch.stack([p, p]).expand(B, 2, V)            # bonus slot irrelevant for the first emitted token
    n, nxt = SP.accept_reject(pp, q.expand(B, 1, V), d, u, g)
    first = torch.where(n == 1, d[:, 0], nxt)
    emp = torch.bincount(first, minlength=V).double() / B
    assert (emp - p).abs().max() < 0.005
    # acceptance rate = sum min(p, q)
    assert abs(n.double().mean() - SP.overlap(p, q)) < 0.005


def test_residual_formula_exact():
    p = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    q = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    beta = SP.overlap(p, q)
    assert math.isclose(beta, 0.7)
    r = SP.residual(p, q)
    assert torch.allclose(r, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    emit = torch.minimum(p, q) + (1 - beta) * r
    assert torch.allclose(emit, p)


def test_markov_chain_multi_token_is_exact():
    """Speculative sampling of 3 tokens from a Markov target with a Markov draft, gamma = 2: the empirical
    distribution over all 27 sequences matches the target's joint distribution."""
    g = torch.Generator().manual_seed(1)
    V, B, N = 3, 120_000, 3
    Pt = torch.softmax(torch.randn(V, V, generator=g, dtype=torch.float64) * 1.2, -1)
    Qt = torch.softmax(torch.randn(V, V, generator=g, dtype=torch.float64) * 1.2, -1)
    seq = torch.zeros(B, 1, dtype=torch.long)                       # start state 0
    out = [[] for _ in range(B)]
    active = torch.arange(B)
    cur = torch.zeros(B, dtype=torch.long)
    lens = torch.zeros(B, dtype=torch.long)
    gamma = 2
    buf = torch.full((B, N + gamma + 1), -1, dtype=torch.long)
    while bool((lens < N).any()):
        idx = (lens < N).nonzero().squeeze(-1)
        c = cur[idx]
        d1 = torch.multinomial(Qt[c], 1, generator=g).squeeze(-1)
        d2 = torch.multinomial(Qt[d1], 1, generator=g).squeeze(-1)
        q = torch.stack([Qt[c], Qt[d1]], 1)
        p = torch.stack([Pt[c], Pt[d1], Pt[d2]], 1)
        u = torch.rand(len(idx), 2, generator=g, dtype=torch.float64)
        n, nxt = SP.accept_reject(p, q, torch.stack([d1, d2], 1), u, g)
        for j, i in enumerate(idx.tolist()):
            toks = [int(d1[j]), int(d2[j])][:int(n[j])] + [int(nxt[j])]
            L = int(lens[i])
            for t in toks:
                if L < N + gamma + 1:
                    buf[i, L] = t
                    L += 1
            lens[i] = L
            cur[i] = toks[-1]
    seqs = buf[:, :N]
    codes = (seqs * torch.tensor([V * V, V, 1])).sum(-1)
    emp = torch.bincount(codes, minlength=V ** N).double() / B
    exact = torch.zeros(V ** N, dtype=torch.float64)
    for a, b, c in itertools.product(range(V), repeat=3):
        exact[a * V * V + b * V + c] = Pt[0, a] * Pt[a, b] * Pt[b, c]
    assert 0.5 * (emp - exact).abs().sum() < 0.01


def _tiny_lm(seed, vocab=11, layers=2, hidden=32):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=vocab, hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, intermediate_size=64, max_position_embeddings=128)
    return LM(cfg).double().eval()


def test_truncate_cache_matches_fresh_prefix():
    m = _tiny_lm(0)
    x = torch.randint(0, 11, (1, 12))
    c = m.new_cache()
    m(x, cache=c)
    SP.truncate_cache(c, 7)
    a = m(x[:, 7:9], cache=c).logits
    b = m(x[:, :9]).logits[:, 7:9]
    assert torch.allclose(a, b, atol=1e-10)


@pytest.mark.parametrize("kind", ["lm", "hidden"])
def test_speculative_greedy_equals_autoregressive(kind):
    target = _tiny_lm(1)
    if kind == "lm":
        draft = SP.LMDraft(_tiny_lm(2, layers=1))
    else:
        from frontierlab.ttc.eagle import new_head
        draft = SP.HiddenDraft(new_head(target.config, 3).double(), target.model.embed_tokens, target.lm_head)
    prompt = torch.tensor([[1, 4, 2, 7]])
    ref = SP.autoregressive_generate(target, prompt, 20, temperature=0)
    for gamma in (1, 3, 5):
        out, st = SP.speculative_generate(target, draft, prompt, 20, gamma=gamma, temperature=0)
        assert torch.equal(out, ref)
        assert st["accepted"] <= st["proposed"]


def test_speculative_sampling_distribution_on_tiny_lm():
    """Two sampled tokens after a fixed prompt: speculative vs the exact target joint (enumerated)."""
    V = 5
    target, small = _tiny_lm(4, vocab=V), _tiny_lm(5, vocab=V, layers=1)
    prompt = torch.tensor([[1, 3]])
    with torch.no_grad():
        p1 = torch.softmax(target(prompt).logits[0, -1], -1)
        exact = torch.zeros(V * V, dtype=torch.float64)
        for a in range(V):
            p2 = torch.softmax(target(torch.tensor([[1, 3, a]])).logits[0, -1], -1)
            exact[a * V:(a + 1) * V] = p1[a] * p2
    g = torch.Generator().manual_seed(0)
    R = 4000
    counts = torch.zeros(V * V, dtype=torch.float64)
    draft = SP.LMDraft(small)
    for _ in range(R):
        out, _ = SP.speculative_generate(target, draft, prompt, 2, gamma=2, temperature=1.0, generator=g,
                                         track_overlap=False)
        a, b = out[0, 2:].tolist()
        counts[a * V + b] += 1
    assert 0.5 * (counts / R - exact).abs().sum() < 0.05


def test_expected_tokens_formula():
    assert SP.expected_tokens_per_round(0.0, 4) == 1.0
    assert SP.expected_tokens_per_round(1.0, 4) == 5.0
    assert math.isclose(SP.expected_tokens_per_round(0.8, 4), (1 - 0.8 ** 5) / 0.2)
    assert math.isclose(SP.walltime_improvement(0.8, 4, 0.05), (1 - 0.8 ** 5) / 0.2 / 1.2)
    assert SP.best_gamma(0.9, 0.02) > SP.best_gamma(0.5, 0.02)


# --------------------------------------------------------------------------------------------- serving

def test_kv_bytes_match_cache_nbytes():
    """The serving table's KV function equals what the real caches hold (float32 model, positions included)."""
    from frontierlab.attention import accounting as acc
    from frontierlab.model.config import toy
    for cfg in [toy(97), toy(97).with_(attention="mla", extra={"kv_lora_rank": 64, "qk_rope_head_dim": 16}),
                toy(97).with_(attention="local_global", extra={"window": 8, "layer_types": ["local", "local", "local", "global"]})]:
        m = LM(cfg).eval()
        c = m.new_cache()
        with torch.no_grad():
            m(torch.randint(0, 97, (1, 20)), cache=c)
        assert c.nbytes() == acc.cache_bytes(cfg, 20, bytes_per=4)          # Module 3: positions included
        assert SV.kv_bytes_seq(cfg, 20, bytes_per=4) == acc.kv_bytes(cfg, 20, bytes_per=4)
    hyb = toy(97).with_(attention="hybrid", extra={"linear_kind": "gdn", "full_every": 2})
    m = LM(hyb).eval()
    c = m.new_cache()
    with torch.no_grad():
        m(torch.randint(0, 97, (1, 20)), cache=c)
    assert c.nbytes() == acc.m05_cache_bytes(hyb, 20, bytes_per=4, state_bytes=4, include_pos=True)


def test_qwen3_kv_per_token_and_designs():
    kv = SV.kv_bytes_seq(SV.QWEN3_1_7B, 131072)
    assert kv / 131072 == 2 * 28 * 8 * 128 * 2                     # 114,688 bytes per token in bf16
    rows = {r["design"]: r for r in SV.kv_table(SV.stage_d_designs())}
    assert rows["MLA d_c=4d"]["kv_bytes_per_token"] < rows["GQA (as released)"]["kv_bytes_per_token"]
    assert rows["MHA"]["kv_bytes_per_token"] == 2 * rows["GQA (as released)"]["kv_bytes_per_token"]
    # DSA keeps every key (plus indexer keys): it saves compute, not memory
    assert rows["DSA k=2048 (GQA main)"]["kv_bytes_per_token"] > rows["GQA (as released)"]["kv_bytes_per_token"]


def test_decode_flops_agree_with_accounting():
    from frontierlab.attention import accounting as acc
    for name, cfg in SV.baseline0_designs().items():
        for S in (100, 5000):
            ref = acc.m05_decode_flops_per_token(cfg, S) if SV._is_m05(cfg) else                 acc.decode_flops_per_token(cfg, S, mode="absorbed" if cfg.attention == "mla" else "naive")
            assert math.isclose(SV.decode_flops(cfg, S), ref, rel_tol=1e-12), name


def test_roofline_phases():
    hw = SV.HARDWARE["H100-SXM"]
    cfg = SV.QWEN3_1_7B
    assert SV.intensity(cfg, "prefill", T=2048) > hw.ridge > SV.intensity(cfg, "decode", B=1, S=2048)
    t1, t8 = SV.decode_step_time(cfg, 1, 2048, hw), SV.decode_step_time(cfg, 8, 2048, hw)
    assert t8 < 2 * t1                                              # small batches ride on the weight read


def test_simulator_runs_and_chunking_cuts_decode_stalls():
    cfg = SV.QWEN3_1_7B
    reqs = SV.poisson_requests(rate=6.0, n=60, prompt=4096, output=128, seed=0)
    pf = SV.simulate(reqs, cfg, policy="prefill_first", gpus=1)
    ch = SV.simulate(reqs, cfg, policy="chunked", gpus=1, chunk=1024)
    dg = SV.simulate(reqs, cfg, policy="disaggregated", gpus=2)
    for s in (pf, ch, dg):
        assert s["requests"] == 60 and np.isfinite(s["ttft_p50"]) and s["tpot_p50"] > 0
    assert ch["tpot_p90"] < pf["tpot_p90"]


def test_rollout_time_capacity():
    small = SV.rollout_time(SV.QWEN3_1_7B, 256, 512, 4096)
    long = SV.rollout_time(SV.QWEN3_1_7B, 256, 512, 32768)
    assert long["batch"] < small["batch"] and long["seconds"] > 8 * small["seconds"]


# --------------------------------------------------------------------------------------------- quantisation

def test_asym_qdq_error_bound_and_exact_constants():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 64, generator=g, dtype=torch.float64)
    for bits in (2, 4, 8):
        y = KQ.asym_qdq(x, bits, 1, 32)
        rng = x.view(4, 2, 32).amax(-1) - x.view(4, 2, 32).amin(-1)
        step = (rng / (2 ** bits - 1)).repeat_interleave(32, 1)
        assert ((y - x).abs() <= step / 2 + 1e-12).all()
    c = torch.full((2, 32), 3.25, dtype=torch.float64)
    assert torch.equal(KQ.asym_qdq(c, 2, 1, 32), c)


def test_kivi_residual_window_is_exact_and_keys_per_channel():
    g = torch.Generator().manual_seed(0)
    k = torch.randn(1, 2, 100, 16, generator=g, dtype=torch.float64)
    v = torch.randn(1, 2, 100, 16, generator=g, dtype=torch.float64)
    kq, vq = KQ.kivi_qdq(k, v, bits=2, group=32, residual=40)
    assert torch.equal(kq[:, :, 32:], k[:, :, 32:]) and torch.equal(vq[:, :, 32:], v[:, :, 32:])   # 1 group quantised
    assert not torch.equal(kq[:, :, :32], k[:, :, :32])
    # per-channel: each channel of a group takes at most 4 distinct values at 2 bits
    assert all(len(torch.unique(kq[0, 0, :32, c])) <= 4 for c in range(16))


def test_quant_kv_attention_matches_gqa_at_high_bits_and_registers():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=50, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, intermediate_size=64, max_position_embeddings=128)
    base = LM(cfg).double().eval()
    x = torch.randint(0, 50, (2, 40))
    q16 = KQ.with_quant_kv(base, bits=16, group=8, residual=4).double()
    r_base = KQ.decode_nll(base, x, prefill=1)
    r16 = KQ.decode_nll(q16, x, prefill=1, ref=base)
    assert np.allclose(r_base["nll"], r16["nll"], atol=1e-6) and max(r16["kl"]) < 1e-8
    q2 = KQ.with_quant_kv(base, bits=2, group=8, residual=4).double()
    r2 = KQ.decode_nll(q2, x, prefill=1, ref=base)
    assert min(r2["kl"]) > 1e-6
    # no quantisation while the context fits in the residual window
    qw = KQ.with_quant_kv(base, bits=2, group=8, residual=64).double()
    assert max(KQ.decode_nll(qw, x, prefill=1, ref=base)["kl"]) < 1e-12


def test_quantize_weights_reports_bits_and_error():
    torch.manual_seed(0)
    m = _tiny_lm(7, vocab=40, hidden=64)
    w0 = m.model.layers[0].mlp.up_proj.weight.clone()
    emb0 = m.model.embed_tokens.weight.clone()
    info = KQ.quantize_weights_(m, bits=4, group=32)
    assert info["matrices"] == 2 * 7 and info["bits_per_weight"] == 4 + 32 / 32
    assert 0 < info["mean_rel_err"] < 0.2
    assert not torch.equal(w0, m.model.layers[0].mlp.up_proj.weight)
    assert torch.equal(emb0, m.model.embed_tokens.weight)            # embedding and tied head untouched


def test_hf_speculative_greedy_matches_plain_on_tiny_qwen3():
    pytest.importorskip("transformers")
    from frontierlab.ttc import hf_spec as HS
    target, draft, prompts = HS._load(True, "cpu")
    for p in prompts[:2]:
        g = torch.Generator().manual_seed(0)
        ref = HS.hf_plain(target, p, 10, 0.0, g)
        out, st = HS.hf_speculative(target, draft, p, 10, 3, 0.0, torch.Generator().manual_seed(0))
        assert out == ref and st["accepted"] <= st["proposed"]
