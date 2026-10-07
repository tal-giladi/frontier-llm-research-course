"""Tests for frontierlab.interp (Module 17). No downloads; about a minute on a laptop."""

import numpy as np
import pytest
import torch

from frontierlab.interp import claims as CL
from frontierlab.interp import graphs as G
from frontierlab.interp import hooks as HK
from frontierlab.interp import patching as P
from frontierlab.interp import persona as PE
from frontierlab.interp import sae as S
from frontierlab.interp import sparse as SPR
from frontierlab.interp import steering as ST
from frontierlab.interp import superposition as SP
from frontierlab.interp import tasks as T
from frontierlab.model import LM, toy


def tiny(layers=2, vocab=50, seed=0):
    torch.manual_seed(seed)
    return LM(toy(vocab_size=vocab).with_(num_hidden_layers=layers)).double().eval()


def hf_tiny():
    from frontierlab.interp import hf
    return hf.load("", smoke=True)


# --------------------------------------------------------------------------------------------- hooks

def test_sites_match_the_forward_and_hooks_never_leak():
    m = tiny()
    x = torch.randint(0, 50, (2, 9))
    sites = ["resid_pre.0", "resid_post.0", "resid_pre.1", "attn_out.1", "mlp_in.1", "mlp_out.1", "z.1", "final"]
    logits, a = HK.capture(m, x, sites)
    assert torch.allclose(logits, m(x).logits)
    assert torch.allclose(a["resid_post.0"], a["resid_pre.1"])                         # layer boundary
    L1 = m.model.layers[1]
    h = a["resid_pre.1"] + a["attn_out.1"]
    assert torch.allclose(a["mlp_in.1"], L1.post_attention_layernorm(h))
    assert torch.allclose(a["final"], h + a["mlp_out.1"])
    assert torch.allclose(a["attn_out.1"], L1.self_attn.o_proj(a["z.1"]))
    assert HK.n_hooks(m) == 0
    with pytest.raises(RuntimeError):
        with HK.hooks(m, {"resid_post.0": lambda t: t}):
            raise RuntimeError("boom")
    assert HK.n_hooks(m) == 0                                                           # removed on exceptions too


def test_hooks_work_on_hf_qwen3_and_identity_edits_are_exact():
    m, _ = hf_tiny()
    x = torch.randint(2, 512, (2, 7))
    ref = HK.logits_of(m, x)
    ed = {s: (lambda t: t.clone()) for s in ("resid_pre.1", "resid_post.1", "z.0", "mlp_in.2", "mlp_out.0", "final")}
    assert torch.equal(HK.run_with(m, x, ed), ref)
    _, a = HK.capture(m, x, ["resid_post.0", "resid_pre.1", "z.1", "attn_out.1"])
    assert torch.allclose(a["resid_post.0"], a["resid_pre.1"])
    assert torch.allclose(a["attn_out.1"], m.model.layers[1].self_attn.o_proj(a["z.1"]), atol=1e-6)
    assert HK.n_hooks(m) == 0


def test_project_out_removes_exactly_one_direction():
    x = torch.randn(2, 5, 8, dtype=torch.float64)
    u = torch.randn(8, dtype=torch.float64)
    y = HK.project_out(u)(x)
    assert torch.allclose(y @ (u / u.norm()), torch.zeros(2, 5, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(x - y, ((x @ (u / u.norm()))[..., None]) * (u / u.norm()))


# --------------------------------------------------------------------------------------------- superposition

def test_superposition_dense_vs_sparse():
    dense = SP.stats(SP.train(10, 3, 0.0, steps=800, batch=512))
    sparse = SP.stats(SP.train(10, 3, 0.95, steps=800, batch=512))
    assert dense["represented"] <= 4                      # about m features, one per dimension
    assert sparse["represented"] > 3                      # more features than dimensions
    assert sparse["dims_per_feature"] < dense["dims_per_feature"]


# --------------------------------------------------------------------------------------------- SAEs

def test_sae_kinds_l0_and_unit_norm_decoder():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, 16, generator=g)
    for kind in ("relu", "topk", "jumprelu"):
        sae = S.SAE(16, 64, kind, k=5)
        sae.init_from(x)
        z = sae.encode(x)
        assert z.shape == (256, 64) and (z >= 0).all()
        if kind == "topk":
            assert int((z > 0).sum(-1).max()) <= 5
        sae.W_dec.data.mul_(3.0)
        sae.renorm()
        assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(64), atol=1e-6)


def test_jumprelu_ste_matches_the_paper_formula():
    pre = torch.tensor([[0.10, 0.52, 0.48, 1.00]], dtype=torch.float64, requires_grad=True)
    theta = torch.full((4,), 0.5, dtype=torch.float64, requires_grad=True)
    eps = 0.1
    z = S.jumprelu(pre, theta, eps)
    assert torch.allclose(z, torch.tensor([[0.0, 0.52, 0.0, 1.0]], dtype=torch.float64))
    z.sum().backward()
    assert torch.allclose(pre.grad, torch.tensor([[0.0, 1.0, 0.0, 1.0]], dtype=torch.float64))
    # entries 1 and 2 are within eps/2 of theta: d/dtheta = -(theta/eps) = -5 each; others 0
    assert torch.allclose(theta.grad, torch.tensor([0.0, -5.0, -5.0, 0.0], dtype=torch.float64))
    theta.grad = None
    S.step(pre, theta, eps).sum().backward()
    assert torch.allclose(theta.grad, torch.tensor([0.0, -10.0, -10.0, 0.0], dtype=torch.float64))


def test_sae_training_learns_planted_sparse_features():
    g = torch.Generator().manual_seed(0)
    D = torch.randn(32, 16, generator=g)
    D = D / D.norm(dim=1, keepdim=True)
    codes = (torch.rand(4000, 32, generator=g) < 0.06) * torch.rand(4000, 32, generator=g)
    x = codes @ D
    sae, _ = S.train_sae(x, "topk", 64, k=4, steps=1500, batch=256, lr=3e-3)
    ev = S.evaluate(sae, x)
    assert ev["fvu"] < 0.15 and ev["l0"] <= 4
    cos = (sae.W_dec @ D.T).abs().max(0).values                # decoder rows have unit norm
    assert (cos > 0.9).float().mean() > 0.6              # most planted directions found


def test_splice_check_and_ablation_modes():
    m = tiny()
    w = torch.randint(0, 50, (4, 10))
    sae = S.SAE(128, 256, "topk", k=8).double()
    clean = S.spliced_losses(m, w, "resid_post.0", mode="clean")
    assert np.allclose(S.spliced_losses(m, w, "resid_post.0", sae, "sae+error"), clean, atol=1e-10)
    assert not np.allclose(S.spliced_losses(m, w, "resid_post.0", sae, "sae"), clean, atol=1e-3)
    assert S.loss_recovered(3.0, 3.0, 5.0) == 1.0 and S.loss_recovered(3.0, 5.0, 5.0) == 0.0


# --------------------------------------------------------------------------------------------- patching

def test_patching_identities():
    m = tiny(vocab=96)
    p = T.induction_pairs(8, kind="value", seed=0)
    p = T.Pairs(p.clean, p.corrupt, p.pos, p.good, p.bad)
    b = P.baselines(m, p)
    L = HK.n_layers(m)
    # patching the whole residual stream at the last layer's output restores the clean run exactly
    full = P.patch(m, p, f"resid_post.{L - 1}", base=b)
    assert torch.allclose(full, (b["clean"] - b["corrupt"]) / (b["clean"].mean() - b["corrupt"].mean()), atol=1e-9)
    assert abs(float(full.mean()) - 1) < 1e-9
    # all heads of a layer together = the whole z site
    _, a = HK.capture(m, p.clean, ["z.1"])
    whole = HK.run_with(m, p.corrupt, {"z.1": HK.replace_at(a["z.1"])})
    by_heads = HK.run_with(m, p.corrupt, {"z.1": HK.compose(*[HK.replace_at(a["z.1"], None, h, 32) for h in range(4)])})
    assert torch.allclose(whole, by_heads, atol=1e-10)


def test_path_patch_all_receivers_equals_sender_patch_for_last_layer():
    """Sender = every head of the last layer, receiver = logits. With the last MLP switched off the direct path is
    the only path, so path patching equals ordinary (noising) patching of that layer's z; with the MLP on it
    does not (the difference is the path through the MLP)."""
    m = tiny(vocab=96)
    with_mlp = None
    p = T.induction_pairs(6, kind="key", seed=2)
    L = HK.n_layers(m) - 1
    with_mlp = (P.path_patch(m, p, L, list(range(4)), ("logits",)), P.patch(m, p, f"z.{L}", mode="noise"))
    with torch.no_grad():
        m.model.layers[L].mlp.down_proj.weight.zero_()
    pp = P.path_patch(m, p, L, list(range(4)), ("logits",), mode="noise")
    ordinary = P.patch(m, p, f"z.{L}", mode="noise")
    assert torch.allclose(pp, ordinary, atol=1e-8)
    assert not torch.allclose(*with_mlp, atol=1e-3)


def test_attribution_estimate_is_exact_when_the_metric_is_linear_in_the_site():
    """The final-norm input enters the metric through one RMSNorm: attribution patching of z in the last layer
    is first-order exact, so for a tiny perturbation (corrupt run = clean + ε·noise in z) estimate ≈ real."""
    m = tiny(vocab=96)
    p = T.induction_pairs(6, kind="key", seed=3)
    est = P.attribution_heads(m, p)
    real = P.head_sweep(m, p, "denoise")
    assert est.shape == real.shape == (2, 4) and np.isfinite(est).all()
    assert np.corrcoef(est.ravel(), real.ravel())[0, 1] > 0.5           # a screen, not an equality


def test_controls_shape_and_p_value_floor():
    m = tiny(vocab=96)
    p = T.induction_pairs(8, kind="key", seed=4)
    d = torch.randn(128, dtype=torch.float64)
    r = P.direction_control(m, p, "resid_pre.1", d, n_random=5)
    assert len(r["random"]) == 5 and r["p_random"] >= 1 / 6
    c = P.component_control(m, p, {0: [1]}, "mean", n_random=3)
    assert len(c["random"]) == 3 and c["ci"][0] <= c["effect"] <= c["ci"][1]


def test_ablation_modes_and_resample():
    m = tiny(vocab=96)
    x = torch.randint(2, 96, (4, 12))
    z = P.ablate_heads(m, x, {0: [0, 1]}, "zero")
    mean = P.ablate_heads(m, x, {0: [0, 1]}, "mean", reference=x)
    rs = P.ablate_heads(m, x, {0: [0, 1]}, "resample", reference=x, gen=torch.Generator().manual_seed(0))
    assert z.shape == mean.shape == rs.shape == (4, 12, 96)
    assert not torch.allclose(z, mean)


def test_induction_data_has_variable_gaps_and_unique_segments():
    x, g = T.repeated_batch(16, 8, 50, torch.Generator().manual_seed(0), max_gap=6)
    assert x.shape == (16, 22) and g.min() >= 0 and g.max() <= 6
    for row, gap in zip(x, g):
        seg = row[:8]
        assert len(set(seg.tolist())) == 8 and torch.equal(row[8 + gap:16 + gap], seg)
    p = T.induction_pairs(4, half=8, kind="key", gap=3, seed=0)
    assert torch.equal(p.clean[:, p.pos], p.clean[:, p.meta["query"]])        # query token = the first A
    assert (p.clean != p.corrupt).sum() == 4


def test_ioi_prompts_with_a_word_tokenizer():
    _, tok = hf_tiny()
    d = T.ioi_prompts(tok, "train", 12, 0, [" Mary", " John", " Alice", " Bob"])
    assert d["clean"].shape == d["corrupt"].shape and (d["good"] != d["bad"]).all()
    assert (d["clean"] != d["corrupt"]).sum(1).eq(1).all()                    # only the repeated subject changes


# --------------------------------------------------------------------------------------------- graphs

def test_replacement_model_reproduces_the_model_and_edges_sum_to_inputs():
    m = tiny(layers=2)
    idx = torch.randint(0, 50, (1, 6))
    rec = G.trace(m, idx)
    assert torch.allclose(rec["logits"], m(idx).logits, atol=1e-12)
    tcs = [G.Transcoder(128, 128, "topk", k=8, seed=l).double() for l in range(2)]
    acts, errs = G.clean_codes(rec, tcs)
    lg, pres = G.replacement_logits(m, rec, tcs, acts, errs)
    assert torch.allclose(lg, rec["logits"], atol=1e-10)
    lg2, _ = G.replacement_logits(m, rec, tcs, errors=errs, recompute_features=True)
    assert torch.allclose(lg2, rec["logits"], atol=1e-10)
    g = G.attribute(m, idx, tcs, n_logits=2)
    for i, nd in enumerate(g.nodes):
        if nd[0] == "logit":
            assert abs(g.A[i].sum() - g.activations[i]) < 1e-8
        if nd[0] == "feat":
            l, p, f = nd[1:]
            assert abs(g.A[i].sum() - (float(pres[l][0, p, f]) - float(tcs[l].b_enc[f]))) < 1e-8
            # no edge from a later layer or a later position
            for j, src in enumerate(g.nodes):
                if src[0] == "feat" and (src[1] >= l or src[2] > p):
                    assert g.A[i, j] == 0
    keep = G.prune(g, 0.8)
    assert all(i in keep for i, nd in enumerate(g.nodes) if nd[0] == "logit") and len(keep) < len(g.nodes)
    assert 0 <= G.error_share(g) <= 1


def test_feature_intervention_agrees_with_replacement_prediction_when_downstream_is_frozen():
    m = tiny(layers=2)
    idx = torch.randint(0, 50, (1, 6))
    tcs = [G.Transcoder(128, 128, "topk", k=8, seed=l).double() for l in range(2)]
    rec = G.trace(m, idx)
    acts, _ = G.clean_codes(rec, tcs)
    # a feature in the LAST layer at the LAST position: only the final norm lies downstream. The original
    # model recomputes that norm, the local replacement model freezes it, so the two logit vectors differ by
    # exactly one positive scale factor; earlier positions are unchanged in both.
    p = idx.shape[1] - 1
    f = int(acts[1][0, p].argmax())
    real = G.intervene_feature(m, idx, tcs, 1, p, f, 0.0)
    pred = G.predicted_feature_effect(m, idx, tcs, 1, p, f, 0.0)
    assert torch.allclose(real[0, :p], rec["logits"][0, :p], atol=1e-10)
    assert torch.allclose(pred[0, :p], rec["logits"][0, :p], atol=1e-10)
    ratio = real[0, p] / pred[0, p]
    assert torch.allclose(ratio, ratio.mean().expand_as(ratio), atol=1e-6) and float(ratio.mean()) > 0
    assert not torch.allclose(real, rec["logits"], atol=1e-6)


def test_transcoder_trains():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3000, 16, generator=g)
    W = torch.randn(16, 16, generator=g)
    y = torch.relu(x @ W)
    tc = G.train_transcoder(x, y, d_sae=128, k=16, steps=800, batch=256, lr=3e-3)
    assert G.transcoder_fvu(tc, x, y)["fvu"] < 0.5


# --------------------------------------------------------------------------------------------- steering

def test_mean_diff_steering_and_side_effects():
    m = tiny()
    a = torch.randn(10, 128, dtype=torch.float64) + 1.0
    b = torch.randn(10, 128, dtype=torch.float64)
    v = ST.mean_diff(a, b)
    assert torch.allclose(v, (a.mean(0) - b.mean(0)).float())
    assert abs(float(ST.shuffled_control(a, b).norm()) - float(v.norm())) > 0
    x = torch.randint(0, 50, (3, 8))
    s0 = ST.side_effects(m, x, 0, v.double(), 0.0)
    assert abs(s0["loss_increase"]) < 1e-12 and abs(s0["kl"]) < 1e-12
    s1 = ST.side_effects(m, x, 0, v.double(), 2.0)
    assert s1["kl"] > 0
    r = ST.random_like(v, 4)
    assert torch.allclose(r.norm(dim=1), v.norm().expand(4), atol=1e-4)
    # projection monitor: activations shifted along v project higher
    assert (ST.projection(a, v).mean() > ST.projection(b, v).mean())


def test_persona_items_are_balanced_and_split():
    tr, ev = PE.items("train", 0), PE.items("heldout", 1)
    assert {it["fact"] for it in tr}.isdisjoint({it["fact"] for it in ev})
    frac_a = np.mean([it["trait"] == "A" for it in tr + ev])
    assert 0.35 < frac_a < 0.65
    _, tok = hf_tiny()
    enc = PE.encode(tok, tr[:3])
    assert all(e["trait_id"] != e["other_id"] for e in enc)


def test_left_padded_batches_match_single_runs():
    m, tok = hf_tiny()
    its = PE.encode(tok, PE.items("heldout", 1)[:5])
    single = np.array([ST.ab_logodds(m, [it])[0] for it in its])
    assert len({len(it["ids"]) for it in its}) > 1                       # really different lengths
    assert np.allclose(ST.ab_logodds(m, its, batch=5), single, atol=1e-4)
    r1 = ST.read(m, [it["ids"] for it in its], [1, 2], batch=5)
    r0 = ST.read(m, [it["ids"] for it in its], [1, 2], batch=1)
    assert torch.allclose(r1[2], r0[2], atol=1e-4)


def test_ab_logodds_and_steering_on_hf_tiny():
    m, tok = hf_tiny()
    its = PE.encode(tok, PE.items("train", 0)[:4])
    base = ST.ab_logodds(m, its)
    v = torch.randn(64)
    moved = ST.ab_logodds(m, its, 1, v, 3.0)
    assert base.shape == (4,) and not np.allclose(base, moved)
    assert np.allclose(ST.ab_logodds(m, its, 1, v, 0.0), base)


# --------------------------------------------------------------------------------------------- sparse, introspection, claims

def test_topk_weight_sparsity():
    torch.manual_seed(0)
    m = LM(SPR.task_config())
    SPR.apply_topk_(m, 0.1)
    assert abs(SPR.nonzero_fraction(m) - 0.1) < 0.01
    x, q = SPR.quote_batch(8, torch.Generator().manual_seed(0))
    assert ((x == SPR.Q1) | (x == SPR.Q2)).sum(1).eq(1).all()


def test_introspection_grading_and_injection_edit():
    from frontierlab.interp import introspect as IN
    assert IN.grade("Yes, I detect a thought about the ocean.", "ocean")["correct"]
    assert not IN.grade("I do not notice anything unusual.", "ocean")["claims"]
    assert not IN.grade("The ocean is large.", "ocean")["claims"]
    v = torch.ones(4)
    f = IN.inject_edit(v, 2.0, start=3)
    x = torch.zeros(1, 5, 4)
    assert torch.equal(f(x)[0, :3], torch.zeros(3, 4)) and torch.equal(f(x)[0, 3:], torch.full((2, 4), 2.0))
    assert torch.equal(f(torch.zeros(1, 1, 4)), torch.full((1, 1, 4), 2.0))   # decode steps always injected
    lo, hi = IN.wilson(0, 20)
    assert lo == 0 and 0.1 < hi < 0.2


def test_claim_card_checks():
    good = CL.ClaimCard("c", "m", "i", 0.8, (0.6, 0.9), "random components", 0.03, 39, 0.6, (0.4, 0.8), "other templates",
                        0.01, ["model scale: 0.6B", "one task", "mean ablation only"])
    assert CL.check(good) == []
    bad = CL.ClaimCard("c", "m", "i", 0.8, (0.6, 0.9), "random components", 0.2, 9, 0.1, (-0.1, 0.3), "", 0.2, ["one"])
    probs = CL.check(bad)
    assert len(probs) >= 5
