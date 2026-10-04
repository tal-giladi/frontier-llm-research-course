"""Tests for frontierlab.optim (Module 7): Newton-Schulz, Muon/AdamW, QK-Clip, µP, schedules, stability, wrapper."""

import math

import pytest
import torch

from frontierlab.metrics import read_jsonl
from frontierlab.model import LM, toy
from frontierlab.optim import cost, mup, schedules
from frontierlab.optim.muon import (CUBIC, NS_SCHEDULES, MuonAdamW, newton_schulz, ns_polynomial, orthogonal_polar,
                                    param_groups, split_params, update_scale)
from frontierlab.optim.qkclip import LogitMonitor, QKClip, head_max_logits
from frontierlab.optim.stability import detect_spikes, diagnose
from frontierlab.optim.stabilizers import OptLM
from frontierlab.testing import cache_agreement, causal_check


def svals(X):
    return torch.linalg.svdvals(X.double())


# ------------------------------------------------------------------------------------ Newton-Schulz

@pytest.mark.parametrize("shape", [(32, 32), (16, 48), (48, 16)])
def test_cubic_ns_converges_to_polar_factor(shape):
    """Classic Newton-Schulz in float64 converges to U V^T (the exact orthogonal factor)."""
    torch.manual_seed(0)
    G = torch.randn(*shape, dtype=torch.float64)
    O = newton_schulz(G, [CUBIC] * 40, dtype=torch.float64)
    assert (O - orthogonal_polar(G)).abs().max() < 1e-10
    assert O.shape == G.shape


def test_quintic_singular_values_band_and_same_vectors():
    """Jordan's 5 quintic steps: singular values in a band around 1 (not equal to 1), singular vectors kept."""
    torch.manual_seed(1)
    G = torch.randn(64, 256, dtype=torch.float64)
    O = newton_schulz(G, "quintic5", dtype=torch.float64)
    s = svals(O)
    assert 0.6 < s.min() and s.max() < 1.25
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    D = U.T @ O @ Vh.T                                   # diagonal if the singular vectors are unchanged
    assert (D - torch.diag(torch.diagonal(D))).abs().max() < 1e-9


def test_v4_hybrid_is_nearly_exact():
    torch.manual_seed(2)
    G = torch.randn(64, 128, dtype=torch.float64)
    s = svals(newton_schulz(G, "v4-hybrid", dtype=torch.float64))
    assert (s - 1).abs().max() < 1e-2
    assert len(NS_SCHEDULES["v4-hybrid"]) == 10


def test_polynomial_matches_matrix_iteration():
    """One NS step maps each singular value s to a s + b s^3 + c s^5 (diagonal matrix check)."""
    s = torch.tensor([0.05, 0.3, 0.7, 1.0], dtype=torch.float64)
    X = torch.diag(s) / s.norm()                         # newton_schulz normalises by the Frobenius norm first
    out = newton_schulz(torch.diag(s), "quintic5", dtype=torch.float64)
    ref = X.diagonal()
    for _ in range(5):
        ref = ns_polynomial(ref)
    assert torch.allclose(out.diagonal(), ref, atol=1e-12)


def test_ns_flops_formula_matches_count():
    assert cost.ns_flops((4, 6), 1) == 4 * 16 * 6 + 2 * 64
    assert cost.ns_flops((6, 4), 1) == cost.ns_flops((4, 6), 1)
    m, B = 768, 524288
    assert abs(cost.ns_flops((m, m), 5) / (6 * m * m * B) - cost.jordan_bound(m, B)) < 1e-12


# ----------------------------------------------------------------------------------- optimizer

def test_split_params_keeps_embedding_head_and_vectors_on_adamw():
    model = LM(toy(vocab_size=97))
    sp = split_params(model)
    muon = [n for n, _ in sp["muon"]]
    assert muon and all(".self_attn." in n or ".mlp." in n for n in muon)
    assert not any("embed" in n or "lm_head" in n or "norm" in n for n in muon)
    total = sum(p.numel() for p in model.parameters())
    assert sum(p.numel() for k in sp for _, p in sp[k]) == total


def test_adamw_groups_match_torch_adamw():
    torch.manual_seed(0)
    a = LM(toy(vocab_size=97)).double()
    b = LM(toy(vocab_size=97)).double()
    b.load_state_dict(a.state_dict())
    oa = MuonAdamW(param_groups(a, optimizer="adamw", lr=1e-2, weight_decay=0.1), lr=1e-2)
    dec = [p for n, p in b.named_parameters() if p.dim() >= 2 and "norm" not in n]
    nod = [p for n, p in b.named_parameters() if p.dim() < 2 or "norm" in n]
    ob = torch.optim.AdamW([{"params": dec, "weight_decay": 0.1}, {"params": nod, "weight_decay": 0.0}], lr=1e-2,
                           betas=(0.9, 0.95), eps=1e-8, foreach=False)
    x = torch.randint(0, 97, (2, 16))
    for _ in range(3):
        for m, o in ((a, oa), (b, ob)):
            o.zero_grad()
            m(x, labels=x).loss.backward()
            o.step()
    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert (p - q).abs().max() < 1e-12, n


def test_muon_matches_torch_muon():
    """Same update as torch.optim.Muon (bf16 Newton-Schulz in both, so a bf16-sized tolerance)."""
    torch.manual_seed(0)
    W0 = torch.randn(48, 32)
    grads = [torch.randn(48, 32) for _ in range(3)]
    for adjust, torch_adjust in (("original", "original"), ("match_rms", "match_rms_adamw")):
        p1, p2 = torch.nn.Parameter(W0.clone()), torch.nn.Parameter(W0.clone())
        o1 = MuonAdamW([{"params": [p1], "kind": "muon", "lr": 0.02, "weight_decay": 0.1, "adjust": adjust}])
        o2 = torch.optim.Muon([p2], lr=0.02, weight_decay=0.1, momentum=0.95, nesterov=True, adjust_lr_fn=torch_adjust)
        for g in grads:
            p1.grad, p2.grad = g.clone(), g.clone()
            o1.step()
            o2.step()
        moved = (p2 - W0).abs().max()
        assert (p1 - p2).abs().max() < 0.02 * moved, adjust


def test_update_rms_matching():
    """RMS of an orthogonalised update is ~1/sqrt(max(A, B)); match_rms brings it to ~0.2 for every shape."""
    torch.manual_seed(0)
    for shape in ((64, 64), (32, 128), (256, 64)):
        O = newton_schulz(torch.randn(*shape), "v4-hybrid", dtype=torch.float64)
        r = O.pow(2).mean().sqrt().item()
        assert abs(r - 1 / math.sqrt(max(shape))) < 0.01 / math.sqrt(max(shape))
        assert abs(r * update_scale(shape, "match_rms") - 0.2) < 0.005
    assert update_scale((128, 32), "original") == 2.0 and update_scale((32, 128), "original") == 1.0


def test_lr_scale_multiplies_step():
    p1, p2 = torch.nn.Parameter(torch.ones(4, 4)), torch.nn.Parameter(torch.ones(4, 4))
    o1 = MuonAdamW([{"params": [p1], "kind": "adamw", "lr": 0.1, "lr_scale": 0.5}])
    o2 = MuonAdamW([{"params": [p2], "kind": "adamw", "lr": 0.05}])
    for p, o in ((p1, o1), (p2, o2)):
        p.grad = torch.full((4, 4), 0.3)
        o.step()
    assert torch.equal(p1, p2)


# ----------------------------------------------------------------------------------------- QK-Clip

def _batch(vocab=61, B=2, T=12, seed=0):
    return torch.randint(0, vocab, (B, T), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("kw", [{"qk_norm": False}, {"qk_norm": False, "num_key_value_heads": 4},
                                {"attention": "mla", "extra": {"kv_lora_rank": 32, "qk_rope_head_dim": 16}},
                                {"attention": "mla", "extra": {"kv_lora_rank": 32, "qk_rope_head_dim": 16,
                                                              "q_lora_rank": 24}}])
def test_qkclip_caps_head_logits_at_tau(kw):
    """After QK-Clip, recomputing S_max per head on the same batch gives exactly min(S_max, tau)."""
    import frontierlab.attention  # noqa: F401  (registers mla)
    torch.manual_seed(0)
    model = LM(toy(vocab_size=61).with_(**kw)).double()
    for n, p in model.named_parameters():          # make logits large
        if "q_proj" in n or "k_proj" in n or "q_b_proj" in n or "kv_b_proj" in n:
            p.data.mul_(8.0)
    x = _batch()
    mods = [l.self_attn for l in model.model.layers]
    inputs = {}
    hooks = [m.register_forward_pre_hook(lambda m, a, i=i: inputs.__setitem__(i, (a[0].detach(), a[1])))
             for i, m in enumerate(mods)]
    model(x)
    for h in hooks:
        h.remove()
    before = [head_max_logits(m, *inputs[i]) for i, m in enumerate(mods)]
    tau = float(torch.stack(before).median())
    clip = QKClip(model, tau, monitor=LogitMonitor(model))
    clip.monitor.smax = {i: b for i, b in enumerate(before)}
    clip()
    after = [head_max_logits(m, *inputs[i]) for i, m in enumerate(mods)]   # same layer inputs as before
    for b, a in zip(before, after):
        assert torch.allclose(a, torch.clamp(b, max=tau), rtol=1e-9, atol=1e-9)
    assert clip.last["clipped_heads"] > 0


def test_qkclip_refuses_qk_norm():
    with pytest.raises(ValueError):
        QKClip(LM(toy(vocab_size=61)), 10.0)


def test_logit_monitor_matches_probes():
    from frontierlab.attention import probes
    torch.manual_seed(0)
    model = LM(toy(vocab_size=61))
    mon = LogitMonitor(model)
    mon.active = True
    x = _batch()
    model.train()
    model(x, labels=x).loss.backward()
    smax = mon.take()
    ref = probes.report(model, x, skip=0)["max_logit"]
    assert abs(max(float(v.max()) for v in smax.values()) - ref) < 1e-4


# ---------------------------------------------------------------------------------------------- µP

def test_width_config_and_mup_base_width_equals_sp():
    base = toy(vocab_size=97)
    wide = mup.width_config(base, 256)
    assert wide.num_attention_heads == 8 and wide.head_dim == base.head_dim and wide.intermediate_size == 768
    model = LM(base)
    scales = mup.lr_scales(model, 128)
    assert set(scales.values()) == {1.0}
    torch.manual_seed(0)
    m = LM(base)
    mup.apply_mup(m, base_width=128)
    assert m.lm_head.output_mult == 1.0 and m.lm_head.weight is m.model.embed_tokens.weight


def test_mup_init_and_lr_scales_follow_the_rules():
    torch.manual_seed(0)
    cfg = mup.width_config(toy(vocab_size=97), 512)
    model = OptLM(cfg)
    m = mup.apply_mup(model, base_width=128)
    assert m == 4
    w = model.model.layers[0].mlp.up_proj.weight
    assert abs(w.std().item() - 0.02 / 2) < 0.001                      # 0.02 / sqrt(m)
    assert abs(model.model.embed_tokens.weight.std().item() - 0.02) < 0.001
    assert model.lm_head.output_mult == 0.25
    s = mup.lr_scales(model, 128, "adamw")
    assert s["model.layers.0.self_attn.q_proj.weight"] == 0.25 and s["model.embed_tokens.weight"] == 1.0
    assert s["model.layers.0.input_layernorm.weight"] == 1.0
    sm = mup.lr_scales(model, 128, "muon", "spectral", "match_rms")
    assert sm["model.layers.0.self_attn.q_proj.weight"] == 0.5
    sd = model.state_dict()
    assert "lm_head.weight" in sd                                       # checkpoint names unchanged


# ------------------------------------------------------------------------------------------- stabilizers

def test_softcap_kind_passes_suite_and_reduces_to_gqa():
    import frontierlab.optim  # noqa: F401
    torch.manual_seed(0)
    cfg = toy(vocab_size=61)
    base = LM(cfg)
    capped = LM(cfg.with_(attention="gqa-softcap", extra={"attn_softcap": 1e6}))
    capped.load_state_dict(base.state_dict())
    x = _batch()
    assert (capped(x).logits - base(x).logits).abs().max() < 1e-5      # huge cap = no cap
    soft = LM(cfg.with_(attention="gqa-softcap", extra={"attn_softcap": 5.0, "clip_qkv": 1.0}))
    assert causal_check(soft, 61) < 1e-9
    assert cache_agreement(soft, 61, chunk=3) < 1e-9


def test_optlm_z_loss_and_softcap():
    torch.manual_seed(0)
    cfg = toy(vocab_size=61)
    plain, z = LM(cfg), OptLM(cfg.with_(extra={"z_loss": 1e-2}))
    z.load_state_dict(plain.state_dict())
    x = _batch()
    lp, lz = plain(x, labels=x), z(x, labels=x)
    lse = torch.logsumexp(lp.logits[:, :-1], -1)
    assert torch.allclose(lz.loss, lp.loss + 1e-2 * lse.pow(2).mean(), atol=1e-6)
    assert torch.equal(lz.per_token_loss, lp.per_token_loss)
    z.eval()
    assert torch.allclose(z(x, labels=x).loss, lp.loss)                  # no z-loss at evaluation
    cap = OptLM(cfg.with_(extra={"final_softcap": 2.0}))
    assert cap(x).logits.abs().max() <= 2.0


# -------------------------------------------------------------------------------------------- schedules

def test_wsd_branch_equals_constant_before_decay():
    for s in range(0, 100):
        c = schedules.lr_at(s, 1000, 1e-3, 10, "constant")
        w = schedules.lr_at(s, 1000, 1e-3, 10, "wsd", decay_start=100, decay_steps=50)
        assert c == w
    vals = [schedules.lr_at(s, 150, 1.0, 10, "wsd", decay_start=100, decay_steps=50, shape="1-sqrt") for s in range(100, 150)]
    assert vals[0] < 1.0 and all(a > b for a, b in zip(vals, vals[1:])) and vals[-1] == 0.0
    from frontierlab.train.loop import lr_at as loop_lr
    assert all(schedules.lr_at(s, 300, 3e-3, 50, "cosine") == loop_lr(s, 300, 3e-3, 50, "cosine") for s in range(300))


def test_branch_cost():
    plan = schedules.branch_plan([1000, 2000, 4000], 0.1)
    c = schedules.branch_cost(plan)
    assert c["wsd_steps"] == 3600 + 100 + 200 + 400 and c["cosine_steps"] == 7000


# ----------------------------------------------------------------------------------------- forensics

def test_detect_spikes_and_diagnose_data_spike():
    steps = list(range(1, 201))
    loss = [5 - 0.005 * s for s in steps]
    for i in (119, 120, 121):                                           # steps 120-122: three bad batches
        loss[i] += 2.0
    metrics = [{"split": "train", "step": s, "loss": l, "grad_norm": 1.0 + (5.0 if 120 <= s <= 122 else 0.0)}
               for s, l in zip(steps, loss)]
    stab = [{"split": "stability", "step": s, "max_logit": 8.0, "ratio_max": 0.01} for s in steps]
    sp = detect_spikes(steps, loss)
    assert len(sp) == 1 and sp[0]["start"] == 120 and sp[0]["end"] == 122
    assert diagnose(metrics, stab)["verdict"] == "data"
    stab_logit = [{**r, "max_logit": 8.0 * (1 + r["step"] / 20)} for r in stab]
    assert diagnose(metrics, stab_logit)["verdict"] == "logit growth"
    stab_lr = [{**r, "ratio_max": 0.1 if r["step"] >= 120 else 0.01} for r in stab]
    assert diagnose(metrics, stab_lr)["verdict"] == "optimizer"
    flat = [{**r, "loss": 5 - 0.005 * r["step"]} for r in metrics]
    assert diagnose(flat, stab)["verdict"] == "none"
    assert diagnose(flat, stab_logit)["verdict"] == "logit growth (no spike)"


# ------------------------------------------------------------------------------- wrapper and resume

ARGS = ["--preset", "toy", "--batch", "4", "--seq", "32", "--lr", "3e-3", "--warmup", "5", "--steps", "16",
        "--log-every", "1", "--eval-every", "100", "--eval-windows", "4", "--ckpt-every", "100", "--device", "cpu"]


@pytest.mark.parametrize("extra", [["--optimizer", "muon"],
                                   ["--optimizer", "muon", "--qk-norm", "off", "--qk-clip", "5", "--stability-log"],
                                   ["--optimizer", "adamw", "--width", "64", "--mup-base-width", "128", "--z-loss", "1e-4"]])
def test_exact_resume_with_module7_wrapper(tiny_data, tmp_path, extra):
    """Straight run == stop at step 7 and resume: bit-identical weights and identical logged losses."""
    from frontierlab.optim.train import main
    a = main(["--run", str(tmp_path / "a"), "--data", str(tiny_data), *ARGS, *extra])
    main(["--run", str(tmp_path / "b"), "--data", str(tiny_data), *ARGS, *extra, "--stop-after", "7"])
    b = main(["--run", str(tmp_path / "b"), "--data", str(tiny_data), *ARGS, *extra])
    for (n, p), (_, q) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(p, q), n
    la = [r["loss"] for r in read_jsonl(tmp_path / "a" / "metrics.jsonl") if r["split"] == "train"]
    lb = [r["loss"] for r in read_jsonl(tmp_path / "b" / "metrics.jsonl") if r["split"] == "train"]
    assert la == lb and len(la) == 16
    if "--stability-log" in extra:
        rows = read_jsonl(tmp_path / "a" / "stability.jsonl")
        assert [r["step"] for r in rows] == list(range(1, 17))
        assert all(r["max_logit"] is not None and "ratio_muon_median" in r for r in rows)


def test_wrapper_run_card_and_branch(tiny_data, tmp_path):
    from frontierlab.optim.train import main
    from frontierlab.runcard import read_run_card
    common = ["--data", str(tiny_data), *ARGS[:-12], "--log-every", "1", "--eval-every", "100", "--eval-windows", "4",
              "--device", "cpu", "--optimizer", "muon"]
    main(["--run", str(tmp_path / "stable"), *common, "--steps", "12", "--schedule", "constant", "--stop-after", "8"])
    card = read_run_card(tmp_path / "stable")
    assert card["optim"]["optimizer"] == "muon" and card["budget"]["optimizer_flops"] > 0
    # branch at step 8: copy of the stable checkpoint, decay over 4 steps
    main(["--run", str(tmp_path / "branch"), *common, "--steps", "12", "--schedule", "wsd", "--decay-start", "8",
          "--decay-steps", "4", "--branch-from", str(tmp_path / "stable" / "checkpoint.pt")])
    rows = [r for r in read_jsonl(tmp_path / "branch" / "metrics.jsonl") if r["split"] == "train"]
    assert [r["step"] for r in rows] == [9, 10, 11, 12]
    assert rows[0]["lr"] < 3e-3 and rows[-1]["lr"] == 0.0
    assert read_run_card(tmp_path / "branch")["parent_run"] == "stable"
