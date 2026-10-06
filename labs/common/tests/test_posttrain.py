"""Module 12: post-training foundations and Eval Suite v2. CPU only, no downloads, about a minute."""

import itertools
import json
import math

import numpy as np
import pytest
import torch

from frontierlab.evals.suite_v2 import compare, pass_at_k
from frontierlab.evals.suite_v2 import ifeval as IF
from frontierlab.model import LM
from frontierlab.posttrain import advantages as A
from frontierlab.posttrain import gsm8k
from frontierlab.posttrain import kl as K
from frontierlab.posttrain import losses as Lo
from frontierlab.posttrain import reward as RW
from frontierlab.posttrain.policy import response_mask, sample, token_logprobs
from frontierlab.posttrain.sft import policy_config, save_policy, sft_loss
from frontierlab.posttrain.tasks import (LetterWorld, Problem, encode_prompts, encode_sft, follows_instruction,
                                         lenient_verify, last_number_verify, make_problems, split_problems,
                                         strict_verify)
from frontierlab.posttrain.tokenizer import EOS, PAD, TOK


# --------------------------------------------------------------------------- tasks and verifiers

def test_tokenizer_and_targets():
    assert TOK.decode(TOK.encode("07+35=") + [EOS, 5]) == "07+35="
    p = Problem(".", 7, "+", 35, 2)
    assert p.prompt == ".07+35=" and p.target == "42"
    assert Problem("P", 7, "+", 35, 2).target == "042"
    assert Problem("Q", 7, "+", 35, 2).target == "#42#"
    assert Problem("E", 50, "-", 8, 2).target == "42!"


def test_split_is_disjoint_and_problems_respect_it():
    train, held = split_problems(2)
    assert not train & held and len(train) + len(held) == 100 * 100 + 5050
    probs = make_problems(500, 2, "+-", ".PQE", seed=3, exclude=held)
    assert all((p.a, p.op, p.b) not in held for p in probs)
    assert all(p.a >= p.b for p in probs if p.op == "-")


def test_verifiers_on_probe_responses():
    p = Problem(".", 7, "+", 35, 2)
    cases = {("42", True): (1, 1, 1), ("42", False): (0, 1, 1), ("4242", True): (0, 0, 1),
             ("1424", True): (0, 0, 1), ("17 42", True): (0, 1, 1), ("042", True): (0, 1, 1), ("41", True): (0, 0, 0)}
    for (text, fin), want in cases.items():
        got = (strict_verify(text, fin, p), last_number_verify(text, fin, p), lenient_verify(text, fin, p))
        assert tuple(map(int, got)) == want, (text, fin, got)


def test_instruction_checks_are_format_only():
    q = Problem("Q", 7, "+", 35, 2)
    assert follows_instruction("#41#", True, q) and not follows_instruction("41", True, q)
    assert not follows_instruction("#42#", False, q)
    assert follows_instruction("123", True, Problem("P", 7, "+", 35, 2))
    assert not follows_instruction("42", True, Problem("P", 7, "+", 35, 2))


# --------------------------------------------------------------------------- policy, masking, SFT

def test_response_mask_includes_eos_and_drops_padding():
    r = torch.tensor([[5, 6, EOS, PAD, PAD], [5, EOS, 7, EOS, 8], [5, 6, 7, 8, 9]])
    mask, fin = response_mask(r)
    assert mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]]
    assert fin.tolist() == [True, True, False]


def test_sampler_logprobs_match_teacher_forced():
    torch.manual_seed(0)
    m = LM(policy_config())
    probs = make_problems(6, 2, seed=1)
    ro = sample(m, encode_prompts(probs), 6, 1.0, torch.Generator().manual_seed(0))
    lp = token_logprobs(m, ro.tokens, ro.prompt_len)
    assert torch.allclose(lp * ro.mask, ro.sampler_logp, atol=1e-5)


def test_sft_loss_ignores_prompt_and_padding():
    torch.manual_seed(0)
    m = LM(policy_config())
    ids, mask = encode_sft(make_problems(4, 2, tags=".PQE", seed=0), max_len=14)
    base = sft_loss(m, ids, mask)
    ids2 = ids.clone()
    ids2[mask == 0] = torch.where(ids2[mask == 0] == PAD, torch.tensor(7), ids2[mask == 0])   # junk in padding
    assert torch.allclose(sft_loss(m, ids2, mask), base, atol=1e-6)


# --------------------------------------------------------------------------- advantages and KL

def test_group_advantages_by_hand():
    r = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    mean = A.group_advantages(r, "mean", "none")
    assert torch.allclose(mean[0], torch.tensor([0.5, -0.5, -0.5, 0.5]))
    loo = A.group_advantages(r, "loo", "none")
    assert torch.allclose(loo, mean * 4 / 3)
    std = A.group_advantages(r, "mean", "group")
    s = math.sqrt(1 / 3)                                  # sample std (ddof=1) of [1,0,0,1]
    assert torch.allclose(std[0], torch.tensor([0.5, -0.5, -0.5, 0.5]) / (s + 1e-6), atol=1e-5)
    assert torch.all(std[1] == 0) and A.zero_variance(r).tolist() == [False, True]
    assert A.effective_fraction(r) == 0.5


def test_gae_matches_hand_computation():
    r, v = torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.2, 0.4, 0.5, 0.0])
    adv = A.gae(r, v, gamma=1.0, lam=0.5)
    d = [0.2, 0.1, 0.5]
    assert torch.allclose(adv, torch.tensor([d[0] + 0.5 * d[1] + 0.25 * d[2], d[1] + 0.5 * d[2], d[2]]))


def test_kl_estimator_gradients_by_enumeration():
    torch.manual_seed(0)
    for _ in range(3):
        out = K.exact_categorical(torch.randn(5), torch.randn(5))
        assert out["k1"].abs().max() < 1e-12                                  # zero in expectation
        assert torch.allclose(out["k2"], out["grad_reverse_kl"], atol=1e-12)  # the intended gradient
        assert torch.allclose(out["k3"], out["grad_forward_kl"], atol=1e-12)  # the other direction
        assert not torch.allclose(out["k3"], out["grad_reverse_kl"], atol=1e-3)


def test_kl_estimator_values_are_unbiased_where_claimed():
    torch.manual_seed(0)
    pi, ref = torch.log_softmax(torch.randn(6), -1).double(), torch.log_softmax(torch.randn(6), -1).double()
    p = pi.exp()
    est = K.estimators(pi, ref)
    true = (p * (pi - ref)).sum()
    assert torch.allclose((p * est["k1"]).sum(), true) and torch.allclose((p * est["k3"]).sum(), true)
    assert (est["k3"] >= 0).all() and (est["k2"] >= 0).all()


# --------------------------------------------------------------------------- losses

def test_aggregation_by_hand():
    x = torch.tensor([[1.0, 1.0, 1.0, 1.0], [4.0, 0.0, 0.0, 0.0]])
    m = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
    assert Lo.aggregate(x, m, "token_mean") == pytest.approx(8 / 5)
    assert Lo.aggregate(x, m, "seq_mean_token_mean") == pytest.approx((1 + 4) / 2)
    assert Lo.aggregate(x, m, "seq_mean_token_sum_norm", norm_len=4) == pytest.approx(8 / 8)
    assert Lo.aggregate(x, m, "prompt_mean", group_size=2) == pytest.approx(8 / 5)
    x4, m4 = torch.cat([x, x]), torch.cat([m, m * 0 + torch.tensor([1.0, 1, 0, 0])])
    assert Lo.aggregate(x4, m4, "prompt_mean", group_size=2) == pytest.approx((8 / 5 + 6 / 4) / 2)


def test_length_bias_of_per_sequence_normalisation():
    m = torch.tensor([[1.0, 1, 0, 0, 0, 0, 0, 0], [1.0] * 8])
    w = Lo.token_weights(m, "seq_mean_token_mean")
    assert float(w[0, 0]) == pytest.approx(1 / 4) and float(w[1, 0]) == pytest.approx(1 / 16)     # short tokens weigh 4x more
    w = Lo.token_weights(m, "token_mean")
    assert float(w[0, 0]) == pytest.approx(1 / 10) and float(w[1, 0]) == pytest.approx(1 / 10)
    w = Lo.token_weights(m, "seq_mean_token_sum_norm", norm_len=8)
    assert float(w[0, 0]) == pytest.approx(1 / 16) and float(w[1, 7]) == pytest.approx(1 / 16)


def test_clipping_and_gradient():
    logp = torch.log(torch.tensor([[0.5, 0.5]])).requires_grad_(True)
    old = torch.log(torch.tensor([[0.25, 0.5]]))                      # ratio 2 and 1
    obj, clipped = Lo.clipped_surrogate(Lo.token_ratio(logp, old), torch.tensor([1.0]), 0.2, 0.28)
    assert torch.allclose(obj, torch.tensor([[1.28, 1.0]])) and clipped.tolist() == [[1.0, 0.0]]
    obj.sum().backward()
    assert logp.grad[0, 0] == 0 and logp.grad[0, 1] == pytest.approx(1.0)   # no gradient through clipped token
    obj_neg, _ = Lo.clipped_surrogate(torch.tensor([[2.0]]), torch.tensor([-1.0]), 0.2, 0.2)
    assert obj_neg.item() == pytest.approx(-2.0)                         # negative advantage: unclipped is the min


def test_sequence_ratio_is_geometric_mean():
    logp, old = torch.log(torch.tensor([[0.5, 0.2, 0.9]])), torch.log(torch.tensor([[0.25, 0.4, 0.9]]))
    m = torch.tensor([[1.0, 1.0, 0.0]])
    s = Lo.sequence_ratio(logp, old, m)
    assert torch.allclose(s, torch.full((1, 3), math.sqrt(2 * 0.5)))


def test_overlong_penalty_dapo_numbers():
    L = torch.tensor([10000.0, 16384.0, 18432.0, 20480.0, 22000.0])     # DAPO: L_max 20,480, L_cache 4,096
    pen = Lo.soft_overlong_penalty(L, l_max=20480, l_cache=4096)
    assert torch.allclose(pen, torch.tensor([0.0, 0.0, -0.5, -1.0, -1.0]))
    mask = torch.ones(2, 3)
    assert Lo.overlong_filter(mask, torch.tensor([True, False])).tolist() == [[1, 1, 1], [0, 0, 0]]


def test_tis_weight_caps():
    w = Lo.tis_weight(torch.log(torch.tensor([0.5, 0.5])), torch.log(torch.tensor([0.1, 0.5])), cap=2.0)
    assert torch.allclose(w, torch.tensor([2.0, 1.0]))


# --------------------------------------------------------------------------- rewards and best-of-n

def test_bt_loss_and_bon_kl():
    assert RW.bt_loss(torch.tensor([1.0]), torch.tensor([0.0])).item() == pytest.approx(math.log(1 + math.e ** -1))
    assert RW.bon_kl(1) == pytest.approx(0.0)
    assert RW.bon_kl(4) == pytest.approx(math.log(4) - 0.75)


def test_bon_expected_matches_enumeration():
    rng = np.random.default_rng(0)
    proxy, val = rng.normal(size=9), rng.normal(size=9)
    for n in (1, 2, 4, 9):
        brute = np.mean([val[list(c)][np.argmax(proxy[list(c)])] for c in itertools.combinations(range(9), n)])
        assert RW.bon_expected(proxy, val, n) == pytest.approx(brute, abs=1e-10)


def test_ece_and_gao_fit():
    p = np.array([0.9] * 10 + [0.6] * 10)
    y = np.array([1] * 9 + [0] + [1] * 3 + [0] * 7, dtype=float)
    assert RW.ece(p, y) == pytest.approx(0.5 * 0.0 + 0.5 * 0.3)
    d = np.linspace(0.2, 3, 20)
    fit = RW.fit_gao(d, d * (1.5 - 0.4 * d), "bon")
    assert fit["alpha"] == pytest.approx(1.5) and fit["beta"] == pytest.approx(0.4)
    assert fit["d_star"] == pytest.approx(1.5 / 0.8)


def test_letter_world_gold_by_hand():
    w = LetterWorld()
    y = "aab"
    want = w.value[0] + w.value[1] + 0.4 * 3
    assert w.gold(y) == pytest.approx(want)
    long = "a" * 10
    assert w.gold(long) == pytest.approx(w.value[0] + 0.4 * 8 - 0.8 * 2)


def test_reward_model_reads_eos_position():
    from frontierlab.posttrain.tasks import encode_strings
    torch.manual_seed(0)
    rm = RW.RewardModel()
    ids, last = encode_strings(["abc", "abcdef"], 16)
    s = rm(ids, last)
    ids2 = ids.clone()
    ids2[0, last[0] + 1:] = 5                                # change tokens after EOS
    assert torch.allclose(rm(ids2, last), s, atol=1e-6)


# --------------------------------------------------------------------------- Eval v2

def test_pass_at_k_matches_brute_force():
    for n, c, k in [(8, 0, 4), (8, 1, 1), (8, 3, 4), (10, 9, 2), (5, 5, 5)]:
        brute = np.mean([any(i < c for i in s) for s in itertools.combinations(range(n), k)])
        assert pass_at_k(n, c, k) == pytest.approx(brute)


def test_compare_verdicts():
    pins = {"x": 1}
    base = {"pins": pins, "components": {"t": {"kind": "task", "items": [0.0] * 50 + [1.0] * 50},
                                          "r": {"kind": "retention", "items": [1.0] * 100}}}
    new = {"pins": pins, "components": {"t": {"kind": "task", "items": [1.0] * 100},
                                         "r": {"kind": "retention", "items": [1.0] * 80 + [0.0] * 20}}}
    c = compare(base, new, default_guard=0.02, n_boot=2000)
    assert c["t"]["verdict"] == "improved" and c["r"]["verdict"] == "regressed" and not c["_passes"]
    with pytest.raises(ValueError):
        compare(base, {**new, "pins": {"x": 2}})


def test_ifeval_checkers():
    assert IF.check("punctuation:no_comma", "no commas here", {})
    assert not IF.check("punctuation:no_comma", "a, b", {})
    assert IF.check("length_constraints:number_words", "one two three", {"relation": "at least", "num_words": 3})
    assert not IF.check("length_constraints:number_words", "one two three", {"relation": "less than", "num_words": 3})
    assert IF.check("change_case:english_lowercase", "all lower", {})
    assert IF.check("startend:quotation", "\"quoted\"", {})
    assert IF.check("detectable_format:json_format", "```json\n{\"a\": 1}\n```", {})
    assert IF.check("detectable_format:title", "<<My Title>>\ntext", {})
    resp = "Sure, here it is:\n\"quoted\""
    assert not IF.check("startend:quotation", resp, {}) and IF.check("startend:quotation", resp, {}, loose=True)
    item = {"instruction_id_list": ["punctuation:no_comma", "change_case:english_lowercase"], "kwargs": [{}, {}]}
    s = IF.score_item(item, "hello World")
    assert s["prompt_strict"] == 0.0 and s["inst_strict"] == [True, False]


def test_gsm8k_extraction():
    gold = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n#### 1,072"
    assert gsm8k.gold_answer(gold) == "1072"
    assert gsm8k.strict_reward("so 1000 + 72 = 1072.\n#### 1072\n\nQuestion: next", "1072") == 1.0
    assert gsm8k.strict_reward("The answer is 1072.", "1072") == 0.0
    assert gsm8k.last_number_reward("The answer is 1072.", "1072") == 1.0
    assert gsm8k.strict_reward("#### 5\n\nQuestion: q\nAnswer: #### 1072", "1072") == 0.0     # cut at the stop


# --------------------------------------------------------------------------- HF adapter and the loop

def test_hf_left_padding_logprobs_match_unpadded():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from frontierlab.posttrain.hf import hf_token_logprobs, left_pad
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(Qwen3Config(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                     max_position_embeddings=64)).eval()
    prompts, resp = [[5, 6, 7], [8, 9, 10, 11, 12]], torch.tensor([[13, 14], [15, 16]])
    pid, patt = left_pad(prompts, 0)
    ids, att = torch.cat([pid, resp], 1), torch.cat([patt, torch.ones_like(resp)], 1)
    padded = hf_token_logprobs(m, ids, att, pid.shape[1])
    for i, p in enumerate(prompts):
        x = torch.tensor([p + resp[i].tolist()])
        alone = hf_token_logprobs(m, x, torch.ones_like(x), len(p))
        assert torch.allclose(padded[i], alone[0], atol=1e-5)


def test_rl_loop_runs_and_resumes_exactly(tmp_path):
    from frontierlab.posttrain.rl import RLConfig, train
    from frontierlab.metrics.jsonl import read_jsonl
    torch.manual_seed(0)
    save_policy(LM(policy_config()), tmp_path / "init.pt")
    kw = dict(init=str(tmp_path / "init.pt"), steps=4, prompts=4, group=4, eval_every=100, ckpt_every=2,
              eval_n=16, kl_beta=0.05, kl_place="loss", entropy_coef=0.01, staleness=1)
    train(RLConfig(run=str(tmp_path / "straight"), **kw))
    train(RLConfig(run=str(tmp_path / "resumed"), **{**kw, "steps": 2}))
    train(RLConfig(run=str(tmp_path / "resumed"), **kw))
    a = [r for r in read_jsonl(tmp_path / "straight" / "metrics.jsonl") if r["split"] == "train"]
    b = [r for r in read_jsonl(tmp_path / "resumed" / "metrics.jsonl") if r["split"] == "train"]
    keys = ("reward", "len", "entropy", "kl_k3", "loss", "grad_norm")
    assert [[r[k] for k in keys] for r in a] == [[r[k] for k in keys] for r in b]
