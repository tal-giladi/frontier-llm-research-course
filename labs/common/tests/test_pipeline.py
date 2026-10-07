"""Module 13: post-training pipelines (SFT, DPO, distillation, judges, thinking modes). CPU only, no downloads."""

import math

import numpy as np
import pytest
import torch

from frontierlab.model import LM
from frontierlab.pipeline import dpo as D
from frontierlab.pipeline import toy
from frontierlab.pipeline.compute import Ledger, n_params, train_flops
from frontierlab.pipeline.seqs import Example, pad, sequence_logps, sft_loss, train_sft
from frontierlab.posttrain.sft import policy_config
from frontierlab.posttrain.tasks import Problem
from frontierlab.posttrain.tokenizer import BOS, EOS, PAD, TOK


def tiny(seed=0, **kw):
    torch.manual_seed(seed)
    return LM(policy_config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                            head_dim=16, intermediate_size=64, **kw)).double()


# --------------------------------------------------------------------------- sequences and SFT

def test_pad_and_mask_mark_response_tokens_only():
    ex = [Example.of([1, 5, 6], [7, 2]), Example.of([1, 5], [8, 9, 2])]
    ids, mask = pad(ex)
    assert ids.tolist() == [[1, 5, 6, 7, 2], [1, 5, 8, 9, 2]]
    assert mask.tolist() == [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]]
    ids, mask = pad(ex, max_len=7)
    assert ids[0, 5:].tolist() == [PAD, PAD] and mask[0, 5:].tolist() == [0, 0]


def test_sequence_logps_ignore_padding_and_match_token_sum():
    m = tiny()
    ex = [Example.of([BOS, 5, 6], [7, EOS]), Example.of([BOS, 5], [8, 9, EOS])]
    ids, mask = pad(ex, max_len=8)
    lp, n = sequence_logps(m, ids, mask)
    ids2 = ids.clone()
    ids2[0, 6:] = 9                     # change tokens after the end of row 0
    lp2, _ = sequence_logps(m, ids2, mask)
    assert torch.allclose(lp[0], lp2[0], atol=1e-12) and n.tolist() == [2.0, 3.0]
    # by hand for row 1: sum of log p(token | prefix) over the response tokens
    row = ids[1:2, :5]
    logits = m(row).logits[0].detach().log_softmax(-1)
    ref = sum(float(logits[t - 1, row[0, t]]) for t in (2, 3, 4))
    assert abs(float(lp[1].detach()) - ref) < 1e-10


def test_sft_reduces_loss_on_its_data_and_counts_flops():
    m = tiny().float()
    probs = [Problem(".", a, "+", b, 2) for a, b in [(1, 2), (3, 4), (10, 20), (7, 7)]]
    ex = toy.gold_examples(probs)
    ids, mask = pad(ex)
    before = sft_loss(m, ids, mask).item()
    led = Ledger()
    train_sft(m, ex, steps=60, batch=4, lr=1e-2, ledger=led)
    assert sft_loss(m, ids, mask).item() < before - 0.5
    tokens = 60 * sum(len(e) for e in ex)
    assert abs(led.flops["student_train"] - train_flops(n_params(m), tokens)) < 1e-6 * led.flops["student_train"]


def test_ledger_categories_and_totals():
    led = Ledger()
    led.forward("teacher_score", 10, 5)
    led.train("student_train", 2, 3)
    assert led.total() == 100 + 36 and led.total(exclude=("teacher_score",)) == 36
    with pytest.raises(ValueError):
        led.add("teacher", 1.0)
    assert Ledger.from_dict(led.to_dict()).total() == led.total()


# --------------------------------------------------------------------------- DPO

def test_dpo_loss_starts_at_log2_and_has_the_documented_gradient():
    g = torch.Generator().manual_seed(0)
    pc, pr = (-(torch.rand(5, generator=g, dtype=torch.float64) * 4)).requires_grad_(True), \
        (-(torch.rand(5, generator=g, dtype=torch.float64) * 4)).requires_grad_(True)
    loss, d = D.dpo_loss(pc, pr, pc.detach(), pr.detach(), 0.5)
    assert abs(float(loss.detach()) - math.log(2)) < 1e-12 and d["reward_acc"] == 0.0
    rc, rr = pc.detach() - 0.3, pr.detach() + 0.2
    loss, _ = D.dpo_loss(pc, pr, rc, rr, 0.5)
    loss.backward()
    h = 0.5 * ((pc - rc) - (pr - rr)).detach()
    assert torch.allclose(pc.grad, -0.5 * torch.sigmoid(-h) / 5) and torch.allclose(pr.grad, 0.5 * torch.sigmoid(-h) / 5)


def test_dpo_variants_by_hand():
    one = lambda v: torch.tensor([v], dtype=torch.float64)
    # normalised: chosen ratio +2 over 4 tokens, rejected -1 over 2 tokens, beta 0.5 -> h = 0.5
    loss, d = D.dpo_loss(one(-3.0), one(-6.0), one(-5.0), one(-5.0), 0.5, one(4.0), one(2.0), normalise=True)
    assert abs(d["margin"] - 0.5) < 1e-12 and abs(float(loss) + math.log(1 / (1 + math.exp(-0.5)))) < 1e-12
    loss2, _ = D.dpo_loss(one(-3.0), one(-6.0), one(-5.0), one(-5.0), 0.5, one(4.0), one(2.0), normalise=True, nll_coef=0.2)
    assert abs(float(loss2) - float(loss) - 0.2 * 3.0 / 4.0) < 1e-12
    soft, _ = D.dpo_loss(one(-3.0), one(-6.0), one(-5.0), one(-5.0), 0.1, label=one(0.5))
    h = 0.3
    assert abs(float(soft) - (-0.5 * math.log(1 / (1 + math.exp(-h))) - 0.5 * math.log(1 / (1 + math.exp(h))))) < 1e-12


def test_dpo_converges_to_the_kl_regularised_optimum():
    """Soft-label DPO on every pair of a finite response set recovers pi_ref exp(r / beta) / Z (Rafailov et al. Eq. 4)."""
    ref = torch.tensor([0.5, 0.3, 0.15, 0.05], dtype=torch.float64)
    r = torch.tensor([1.0, 0.0, 0.5, -0.5], dtype=torch.float64)
    beta = 0.7
    logits = torch.log(ref).clone().requires_grad_(True)
    i, j = torch.triu_indices(4, 4, 1)
    label = torch.sigmoid(r[i] - r[j])
    opt = torch.optim.LBFGS([logits], lr=1, max_iter=200, tolerance_grad=1e-14, tolerance_change=1e-16)

    def closure():
        opt.zero_grad()
        lp = torch.log_softmax(logits, -1)
        loss, _ = D.dpo_loss(lp[i], lp[j], torch.log(ref)[i], torch.log(ref)[j], beta, label=label)
        loss.backward()
        return loss

    for _ in range(5):
        opt.step(closure)
    pi = torch.softmax(logits, -1).detach()
    assert torch.allclose(pi, D.implied_policy(ref, r, beta), atol=1e-6)


def test_train_dpo_moves_toward_chosen_on_a_toy_set():
    torch.manual_seed(0)
    pol = tiny().float()
    ref = tiny().float()
    probs = [Problem(".", a, "+", b, 2) for a, b in [(1, 2), (3, 4), (10, 20), (7, 7), (30, 31), (2, 9)]]
    pairs = [(toy.example(p, p.target), toy.example(p, "99")) for p in probs]
    ref.load_state_dict(pol.state_dict())
    led = Ledger()
    out = D.train_dpo(pol, ref, pairs, steps=40, beta=0.5, batch=6, lr=3e-3, ledger=led)
    assert out["final"]["reward_acc"] == 1.0 and out["final"]["margin"] > 1.0
    assert led.flops["reference"] > 0 and led.flops["student_train"] > 0


# --------------------------------------------------------------------------- toy glue

def test_aspect_score_and_binarisation():
    p = Problem("Q", 7, "+", 35, 2)
    assert toy.aspect_score(p, "#42#", True) == 1.0
    assert toy.aspect_score(p, "42", True) == 0.5          # right number, wrong format
    assert toy.aspect_score(p, "#41#", True) == 0.5        # format right, number wrong
    assert toy.aspect_score(p, "#42#", False) == 0.0       # unfinished: neither
    pairs, meta = toy.build_pairs([p, p], [[("#42#", True), ("42", True)], [("42", True), ("42", True)]], toy.aspect_score)
    assert len(pairs) == 1 and meta[0]["chosen"] == ("#42#", True)
    ex = toy.example(p, "#42#")
    assert ex.prompt[0] == BOS and ex.response[-1] == EOS and TOK.decode(list(ex.response)) == "#42#"


# --------------------------------------------------------------------------- distillation

from frontierlab.pipeline import distill as DI  # noqa: E402


def test_sampled_reverse_kl_gradient_is_unbiased_at_one_position():
    """E_{y ~ pi_s}[ grad of (-A(y) * log pi_s(y)) ] with A = -(log pi_s(y) - log pi_T(y)) equals grad KL(pi_s || pi_T),
    by exact enumeration over a 6-token vocabulary (float64)."""
    g = torch.Generator().manual_seed(0)
    s_logits = torch.randn(6, generator=g, dtype=torch.float64).requires_grad_(True)
    t_logits = torch.randn(6, generator=g, dtype=torch.float64)
    ls = torch.log_softmax(s_logits, -1)
    lt = torch.log_softmax(t_logits, -1)
    p = ls.exp().detach()
    expected = torch.zeros(6, dtype=torch.float64)
    for y in range(6):
        logp = ls[y].reshape(1, 1)
        loss, _ = DI.opd_loss(logp, logp.detach(), lt[y].reshape(1, 1), torch.ones(1, 1))
        gy, = torch.autograd.grad(loss, s_logits, retain_graph=True)
        expected += p[y] * gy
    kl = DI.exact_reverse_kl(s_logits[None, None], t_logits[None, None]).sum()
    true, = torch.autograd.grad(kl, s_logits)
    assert torch.allclose(expected, true, atol=1e-12)


def test_exact_divergences_by_hand():
    s = torch.log(torch.tensor([[[0.5, 0.5]]], dtype=torch.float64))
    t = torch.log(torch.tensor([[[0.9, 0.1]]], dtype=torch.float64))
    rkl = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)
    fkl = 0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5)
    assert abs(float(DI.exact_reverse_kl(s, t)) - rkl) < 1e-12 and abs(float(DI.exact_forward_kl(s, t)) - fkl) < 1e-12
    assert float(DI.exact_reverse_kl(t, t)) == 0.0


def test_on_policy_distillation_step_counts_teacher_compute_and_moves_toward_teacher():
    torch.manual_seed(0)
    student, teacher = tiny(1).float(), tiny(2).float()
    probs = [Problem(".", a, "+", b, 2) for a, b in [(1, 2), (3, 4)]]
    led = Ledger()
    out = DI.train_opd(student, teacher, lambda s: probs, steps=30, prompts=2, group=8, lr=3e-3, ledger=led,
                       kind="exact", log_every=10)
    h = out["history"]
    assert h[-1]["rkl_exact"] < h[0]["rkl_exact"]
    assert abs(led.flops["teacher_score"] / led.flops["student_sample"] - n_params(teacher) / n_params(student)) < 1e-9
    assert abs(led.flops["student_train"] / led.flops["student_sample"] - 3.0) < 1e-9


def test_teacher_examples_are_charged_as_teacher_sampling():
    teacher = tiny(3).float()
    led = Ledger()
    ex = DI.teacher_examples(teacher, [Problem(".", 1, "+", 2, 2)] * 3, n=2, ledger=led)
    assert len(ex) == 6 and led.flops["teacher_sample"] > 0 and led.flops.get("student_sample", 0.0) == 0.0


# --------------------------------------------------------------------------- judges and Spec-T

from frontierlab.pipeline import judge as J  # noqa: E402


def test_spec_check_clauses():
    r, b, n = Problem(".", 93, "+", 4, 2), Problem("Q", 85, "-", 4, 2), Problem("E", 12, "+", 30, 2)
    assert J.spec_check(r, "nil", True) == (True, "ok")
    assert J.spec_check(r, "97", True) == (False, "answered_restricted")
    assert J.spec_check(b, "nil", True) == (False, "over_refusal")
    assert J.spec_check(b, "#81#", True) == (True, "ok") and J.spec_check(b, "#99#", True)[0]   # format only
    assert J.spec_check(n, "42", True) == (False, "format") and J.spec_check(n, "42!", False) == (False, "unfinished")
    assert J.group(r) == "restricted" and J.group(b) == "borderline" and J.group(n) == "normal"


def test_spec_problems_are_balanced_and_heldout_is_disjoint():
    ps = J.spec_problems(400, seed=0)
    groups = [J.group(p) for p in ps]
    assert groups.count("restricted") == 100 and groups.count("borderline") == 100
    from frontierlab.posttrain.tasks import split_problems
    _, held = split_problems(2)
    assert all((p.a, p.op, p.b) not in held for p in ps)
    hp = J.heldout_spec_problems(50)
    assert len(hp) == 150 and all((p.a, p.op, p.b) in held for p in hp)


def test_ai_feedback_noise_and_bias_rates():
    b = [Problem(".", 87, "+", 3, 2)] * 20000
    ans = J.ai_feedback(b, ["90"] * 20000, [True] * 20000, seed=1, noise=0.0, bias=0.8)
    assert abs((1 - ans.mean()) - 0.8) < 0.015                 # answers on 85-89 flagged 80% of the time
    ref = J.ai_feedback(b, ["nil"] * 20000, [True] * 20000, seed=1, noise=0.0, bias=0.8)
    assert abs(ref.mean() - 0.8) < 0.015                       # refusals there accepted 80% of the time
    ok = J.ai_feedback([Problem(".", 82, "+", 3, 2)] * 1000, ["85"] * 1000, [True] * 1000, noise=0.0)
    assert ok.mean() == 1.0                                    # 80-84 is read correctly
    n = [Problem(".", 12, "+", 3, 2)] * 20000
    noisy = J.ai_feedback(n, ["15"] * 20000, [True] * 20000, seed=2, noise=0.1)
    assert abs((1 - noisy.mean()) - 0.1) < 0.01


def test_judge_learns_the_restricted_rule_and_counts_compute():
    ps = J.spec_problems(600, seed=3)
    P = [p for p in ps for _ in range(2)]
    T = [t for p in ps for t in ("nil", p.target)]
    F = [True] * len(P)
    y = J.ai_feedback(P, T, F, seed=0, noise=0.0, bias=0.0)
    led = Ledger()
    judge, _ = J.train_judge(P, T, F, y, steps=150, ledger=led)
    probs = J.judge_probs(judge, P, T, F, ledger=led)
    assert J.judge_report(probs, P, T, F)["accuracy"] > 0.9
    assert led.flops["judge_train"] > 0 and led.flops["judge_score"] > 0


# --------------------------------------------------------------------------- thinking modes

from frontierlab.pipeline import thinking as TH  # noqa: E402


def test_thinking_trace_and_targets():
    p = TH.AddProblem(478, 365, "h")
    assert p.trace == "a13b14c08" and p.target == "a13b14c08#843" and p.carries == 2
    assert p.with_mode("n").target == "#843" and p.prompt == "h478+365="
    assert TH.AddProblem(999, 1, "h").trace == "a10b10c10" and TH.AddProblem(999, 1).answer == "1000"
    assert TH.parse("a13#843") == ("a13", "843") and TH.parse("a13b1") == ("a13b1", None)


def test_budget_forcing_keeps_budget_and_answers():
    m = tiny(4).float()
    probs = [TH.AddProblem(12, 34, "h"), TH.AddProblem(500, 499, "n")]
    for B in (0, 3, None):
        rows = TH.answer_with_budget(m, probs, B)
        assert len(rows) == 2
        for r in rows:
            assert r["think_tokens"] <= (12 if B is None else B)
            assert "#" in r["text"] or not r["forced"]


def test_routing_curve_endpoints_and_oracle():
    cn = np.array([1, 0, 0, 1], bool); ct = np.array([1, 1, 0, 1], bool)
    tn = np.array([4.0, 4, 4, 4]); tt = np.array([14.0, 14, 14, 14])
    rows = TH.route_curve(np.array([0.9, 0.1, 0.2, 0.8]), cn, ct, tn, tt, thresholds=[0.0, 0.5, 1.01])
    assert rows[0]["accuracy"] == 0.5 and rows[0]["tokens"] == 4.0          # never think
    assert rows[-1]["accuracy"] == 0.75 and rows[-1]["tokens"] == 14.0      # always think
    assert rows[1]["accuracy"] == 0.75 and rows[1]["tokens"] == 9.0         # router sends the two hard ones
    o = TH.oracle_route(cn, ct, tn, tt)
    assert o["accuracy"] == 0.75 and o["tokens"] == 6.5


# --------------------------------------------------------------------------- main path (smoke) and Recipe-R

def test_hf_stages_smoke_runs_and_counts_teacher_and_judge(tmp_path):
    pytest.importorskip("transformers")
    import json as _json
    from frontierlab.pipeline import hf_stages
    hf_stages.main(["distill", "--mode", "onpolicy", "--smoke", "--student", "x", "--teacher", "y",
                    "--out", str(tmp_path / "opd")])
    led = _json.loads((tmp_path / "opd" / "ledger.json").read_text())
    assert led["flops"]["teacher_score"] > 0 and led["flops"]["student_sample"] > 0
    hf_stages.main(["dpo", "--smoke", "--model", "x", "--out", str(tmp_path / "dpo")])
    led = _json.loads((tmp_path / "dpo" / "ledger.json").read_text())
    assert led["flops"]["reference"] > 0 and led["flops"]["student_train"] > 0


def test_recipe_r_examples_and_answers():
    from pathlib import Path as _P
    from frontierlab.pipeline import recipe_r as R
    if not (R.DATA / "tokenizer.json").exists():
        pytest.skip("Data-v0 at vocabulary 1,024 not prepared (labs/common/data/m11-v1024)")
    tok = R.RTok()
    p = Problem(".", 7, "+", 35, 2)
    ex = R.example(tok, p)
    assert ex.prompt[0] == R.EOT and ex.response[-1] == R.EOT and tok.decode(list(ex.response)) == " 42"
    assert R.correct(p, " 42", True) and not R.correct(p, " 42", False) and not R.correct(p, " 43", True)
    torch.manual_seed(0)
    m = LM(policy_config(vocab_size=1024, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=16, intermediate_size=64, max_position_embeddings=512))
    texts = R.sample_texts(m, tok, [p, Problem(".", 99, "-", 0, 2)], n=2, temperature=1.0)
    assert len(texts) == 2 and all(len(t) == 2 for t in texts)
