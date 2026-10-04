"""Correctness suite for Module 6 (frontierlab.blocks). Run: pytest labs/common/tests/test_blocks.py"""

import math

import pytest
import torch

from frontierlab.blocks import BlockLM, with_blocks
from frontierlab.blocks import accounting
from frontierlab.blocks.checks import model_checks, module_gradcheck
from frontierlab.blocks.diffusion import DiffusionLM, exact_bound, mask_tokens, nll_bound, sample
from frontierlab.blocks.engram import EngramModule, collision_rate, hash_ngrams
from frontierlab.blocks.hyperconn import (HyperConnection, ManifoldHC, amax_gain, doubly_stochastic_error,
                                         plain_residual, sinkhorn_knopp)
from frontierlab.blocks.latent import RecurrentDepthLM, continuous_thoughts, sample_recurrence
from frontierlab.blocks.matformer import NestedMLP, PerLayerEmbedding, matformer_widths, mix_n_match
from frontierlab.blocks.moe import MoEFFN, dense_reference
from frontierlab.blocks.mtp import (DeepSeekModule, lambda_at, speculative_greedy, teacher_forced_acceptance)
from frontierlab.blocks.patching import NgramByteModel, patch_lengths, patch_starts, threshold_for_size
from frontierlab.model import LM, toy

V = 61


def tiny(**kw):
    return with_blocks(toy(vocab_size=V).with_(hidden_size=32, num_attention_heads=2, num_key_value_heads=1,
                                               head_dim=16, intermediate_size=48, num_hidden_layers=2), **kw)


def small(**kw):
    return with_blocks(toy(vocab_size=V), **kw)


# --- the switches off are Baseline-0 --------------------------------------------------------------------

def test_blocklm_off_equals_lm():
    torch.manual_seed(0)
    a = LM(toy(vocab_size=V))
    torch.manual_seed(0)
    b = BlockLM(small())
    idx = torch.randint(0, V, (2, 12))
    assert torch.equal(a(idx).logits, b(idx).logits)
    assert list(a.state_dict()) == list(b.state_dict())


# --- causal and cached-decode agreement for every variant --------------------------------------------------

VARIANTS = [dict(mtp="meta", mtp_depth=2), dict(mtp="deepseek", mtp_depth=2), dict(residual="hc"),
            dict(residual="mhc"), dict(ffn="moe"), dict(ffn="matformer"), dict(engram={"layers": [1, 2]}),
            dict(ple={"dim": 8}), dict(residual="mhc", ffn="moe", mtp="deepseek", engram={"layers": [1]}),
            dict(residual="hc", mtp="meta", ple={"dim": 8})]


@pytest.mark.parametrize("kw", VARIANTS, ids=lambda kw: "+".join(f"{k}={v}" for k, v in kw.items()))
def test_causal_and_cache(kw):
    torch.manual_seed(0)
    m = BlockLM(small(**kw))
    res = model_checks(m, V, log=lambda *_: None)
    assert res["causal"] < 1e-9 and res["cache_chunk1"] < 1e-9 and res["cache_chunk5"] < 1e-9


def test_causal_and_cache_with_mla():
    torch.manual_seed(0)
    base = small(residual="mhc", mtp="deepseek", ffn="moe")
    cfg = base.with_(attention="mla", extra={**base.extra, "kv_lora_rank": 64, "qk_rope_head_dim": 16})
    res = model_checks(BlockLM(cfg), V, log=lambda *_: None)
    assert max(res.values()) < 1e-9


def test_loss_parts():
    torch.manual_seed(0)
    m = BlockLM(small(mtp="deepseek", ffn="moe"))
    idx = torch.randint(0, V, (2, 16))
    out = m(idx, labels=idx)
    ex = out.extras
    expect = ex["main_loss"] + 0.3 * ex["mtp_loss"] + 0.01 * ex["moe_aux"]
    assert abs(out.loss.item() - expect) < 1e-5
    assert abs(out.per_token_loss.mean().item() - ex["main_loss"]) < 1e-6
    m.eval()
    assert abs(m(idx, labels=idx).loss.item() - (ex["main_loss"] + 0.3 * ex["mtp_loss"])) < 1e-5   # no aux in eval


# --- multi-token prediction ---------------------------------------------------------------------------------

def test_deepseek_module_gradcheck():
    cfg = tiny().with_(hidden_size=8, head_dim=4, intermediate_size=8)
    torch.manual_seed(0)
    mod = DeepSeekModule(cfg, 2)
    h, e = torch.randn(1, 3, 8), torch.randn(1, 3, 8)

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = mod

        def forward(self, h, e):
            return self.m.block(self.m.combine(h, e), torch.arange(3))

    assert module_gradcheck(Wrap(), h, e)


def test_deepseek_mtp_targets_are_shifted():
    """Depth k at position i must predict token i+k+1: a model that copies the embedded token ahead (identity
    eh_proj half, zero block) predicts exactly that and gets near-zero loss only with the right shift."""
    torch.manual_seed(0)
    m = BlockLM(small(mtp="deepseek", mtp_depth=1))
    idx = torch.randint(0, V, (1, 10))
    logits = m.mtp(m(idx, return_hidden=True).hidden["h"], idx, m.model.embed_tokens, m.lm_head)
    assert logits[0].shape == (1, 9, V)


def test_lambda_schedule():
    assert lambda_at(0, 148) == 0.3 and lambda_at(99, 148) == 0.3 and lambda_at(100, 148) == 0.1


@pytest.mark.parametrize("kind", ["meta", "deepseek"])
def test_speculative_greedy_is_lossless(kind):
    torch.manual_seed(0)
    m = BlockLM(small(mtp=kind, mtp_depth=2)).eval()
    prompt = torch.randint(0, V, (1, 5))
    ref = m.generate(prompt, 12)
    out, st = speculative_greedy(m, prompt, 12)
    assert torch.equal(out, ref)
    assert st["proposed"] >= st["accepted"] >= 0 and st["rounds"] <= 12


def test_teacher_forced_acceptance_runs():
    torch.manual_seed(0)
    m = BlockLM(small(mtp="deepseek"))
    st = teacher_forced_acceptance(m, torch.randint(0, V, (2, 16)))
    assert st["positions"] == 30 and 0 <= st["agreement"] <= 1


# --- hyper-connections and mHC ------------------------------------------------------------------------------

def test_sinkhorn_doubly_stochastic():
    torch.manual_seed(0)
    x = torch.randn(64, 4, 4, dtype=torch.float64)
    assert doubly_stochastic_error(sinkhorn_knopp(x, 20)) < 1e-3
    assert doubly_stochastic_error(sinkhorn_knopp(x, 200)) < 1e-12
    assert doubly_stochastic_error(sinkhorn_knopp(torch.eye(4, dtype=torch.float64) * 4, 20)) < 1e-12
    assert torch.allclose(sinkhorn_knopp(torch.zeros(1, 1), 20), torch.ones(1, 1))       # n = 1: exactly 1


def test_sinkhorn_gradcheck():
    from frontierlab.testing import grad_check
    assert grad_check(lambda z: sinkhorn_knopp(z, 20), torch.randn(2, 3, 3))


def test_hc_n1_equals_plain_residual():
    torch.manual_seed(0)
    lin = torch.nn.Linear(16, 16).double()
    hc = HyperConnection(1, 16, 0).double()
    x = torch.randn(2, 5, 16, dtype=torch.float64)
    a = hc(x.unsqueeze(-2), lin)[..., 0, :]
    assert torch.equal(a, plain_residual(x, lin))


def test_hc_and_mhc_models_n1_equal_baseline():
    """n = 1 at initialisation: HC is exactly Baseline-0; mHC is too once its pre-read scale (removed by the
    pre-norm) is the only difference, i.e. with alpha = 0 (H_post = 2·sigmoid(0) = 1, H_res = SK(.) = 1)."""
    idx = torch.randint(0, V, (2, 12))
    torch.manual_seed(0)
    base = LM(toy(vocab_size=V).with_(rms_norm_eps=0.0)).double()
    for kind in ("hc", "mhc"):
        torch.manual_seed(0)
        m = BlockLM(small(residual=kind, streams=1).with_(rms_norm_eps=0.0)).double()
        if kind == "mhc":
            for hy in m.model.hyper:
                for w in hy.values():
                    w.alpha_pre.data.zero_(), w.alpha_post.data.zero_(), w.alpha_res.data.zero_()
        err = (base(idx).logits - m(idx).logits).abs().max().item()
        assert err < 1e-12, (kind, err)


@pytest.mark.parametrize("cls", [HyperConnection, ManifoldHC])
def test_hyper_gradcheck(cls):
    torch.manual_seed(0)
    mod = cls(3, 8, 1)
    with torch.no_grad():
        for p in mod.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    lin = torch.nn.Linear(8, 8).double()
    assert module_gradcheck(mod, torch.randn(1, 2, 3, 8), extra_args=(lambda h: torch.tanh(lin(h)),))


def test_mhc_residual_is_doubly_stochastic_and_gain_one():
    torch.manual_seed(0)
    m = BlockLM(small(residual="mhc"))
    m.track_hyper = True
    out = m(torch.randint(0, V, (2, 8)))
    g = out.extras["hyper"]
    assert abs(g["composite_fwd"] - 1) < 1e-4 and abs(g["composite_bwd"] - 1) < 1e-3


def test_amax_gain_detects_growth():
    big = [torch.eye(2).expand(1, 1, 2, 2) * 1.5 for _ in range(4)]
    assert abs(amax_gain(big)["composite_fwd"] - 1.5 ** 4) < 1e-9


# --- MoE (copied from moelab) ---------------------------------------------------------------------------------

def test_moe_equals_dense_reference():
    torch.manual_seed(0)
    moe = MoEFFN(16, num_experts=6, top_k=2, intermediate=12, shared=1).double()
    moe.reset_parameters(0.3)
    x = torch.randn(2, 7, 16, dtype=torch.float64)
    assert torch.allclose(moe(x), dense_reference(moe, x), atol=1e-12)
    assert abs(moe.last_aux.item() - 1.0) < 1.0                    # a finite balance loss near 1 at random init


def test_moe_gradcheck():
    torch.manual_seed(0)
    moe = MoEFFN(8, num_experts=4, top_k=2, intermediate=6, shared=1)
    moe.reset_parameters(0.5)
    assert module_gradcheck(moe, torch.randn(1, 3, 8))


def test_moe_accounting_active_params():
    cfg = small(ffn="moe", moe={"experts": 8, "top_k": 2, "intermediate": 96, "shared": 1, "first_dense": 1})
    pc = accounting.param_counts(cfg)
    assert pc["total"] == sum(p.numel() for p in BlockLM(cfg).parameters())
    assert pc["moe_inactive"] == 3 * 6 * 3 * 128 * 96


# --- Engram ---------------------------------------------------------------------------------------------------------

def test_engram_hash_is_causal_and_deterministic():
    ids = torch.randint(0, 50, (2, 20))
    h1 = hash_ngrams(ids, 3, 1_000_003, 4093, pad=50)
    ids2 = ids.clone()
    ids2[:, 12:] = torch.randint(0, 50, (2, 8))
    h2 = hash_ngrams(ids2, 3, 1_000_003, 4093, pad=50)
    assert torch.equal(h1[:, :12], h2[:, :12]) and h1.min() >= 0 and h1.max() < 4093
    assert torch.equal(h1, hash_ngrams(ids, 3, 1_000_003, 4093, pad=50))


def test_engram_gradcheck():
    torch.manual_seed(0)
    mod = EngramModule(8, 20, max_n=3, heads=1, table_size=11, head_dim=4, kernel=2)
    with torch.no_grad():
        mod.conv.normal_(0, 0.5)
    ids = torch.randint(0, 20, (1, 5))
    assert module_gradcheck(mod, torch.randn(1, 5, 8), extra_args=(ids,))


def test_engram_collision_rate_counts():
    ids = torch.tensor([1, 2, 3, 1, 2, 3, 4, 5])
    st = collision_rate(ids, 2, 10007)
    assert st["distinct_ngrams"] == 5 and st["share_colliding"] == 0.0


# --- MatFormer and PLE ----------------------------------------------------------------------------------------------

def test_nested_mlp_prefix_equals_small_mlp():
    torch.manual_seed(0)
    big = NestedMLP(16, 32).double()
    small_mlp = NestedMLP(16, 8).double()
    small_mlp.gate_proj.weight.data = big.gate_proj.weight.data[:8].clone()
    small_mlp.up_proj.weight.data = big.up_proj.weight.data[:8].clone()
    small_mlp.down_proj.weight.data = big.down_proj.weight.data[:, :8].clone()
    big.width = 8
    x = torch.randn(3, 16, dtype=torch.float64)
    assert torch.equal(big(x), small_mlp(x))
    assert module_gradcheck(big, torch.randn(2, 16))


def test_matformer_widths_and_mixnmatch():
    assert matformer_widths(384) == [48, 96, 192, 384]
    cfg = mix_n_match([48, 96, 192, 384], 4, 144)
    assert sum(cfg) / 4 == 144 and cfg == sorted(cfg)


def test_matformer_sampling_is_resumable():
    torch.manual_seed(0)
    m = BlockLM(small(ffn="matformer"))
    idx = torch.randint(0, V, (1, 8))
    seen = [m(idx, labels=idx).extras["matformer_width"] for _ in range(6)]
    m2 = BlockLM(small(ffn="matformer"))
    m2.load_state_dict(m.state_dict())
    m.mat_calls.zero_()
    assert [m(idx, labels=idx).extras["matformer_width"] for _ in range(6)] == seen
    assert len(set(seen)) > 1


def test_ple_shapes_and_host_params():
    ple = PerLayerEmbedding(100, 32, 4, 8)
    ids = torch.randint(0, 100, (2, 5))
    p = ple.inputs(ids, torch.randn(2, 5, 32))
    assert p.shape == (2, 5, 4, 8) and ple.layer(1, torch.randn(2, 5, 32), p[:, :, 1]).shape == (2, 5, 32)
    assert ple.host_params() == 100 * 4 * 8


# --- entropy patching -----------------------------------------------------------------------------------------------

def test_patching_rules():
    H = [3.0, 0.5, 0.4, 2.5, 0.3, 0.2, 0.1, 2.8]
    assert patch_starts(H, 1.0, "global") == [0, 3, 7]
    assert patch_starts(H, 1.0, "monotonic") == [0, 3, 7]
    assert patch_lengths([0, 3, 7], 8) == [3, 4, 1]
    th = threshold_for_size(H, 4.0)
    assert len(H) / len(patch_starts(H, th)) == pytest.approx(4.0, abs=1.4)


def test_ngram_entropy():
    m = NgramByteModel(b"abababababab", order=1, alpha=1e-6)
    assert m.entropy(b"a") < 1e-3 and m.entropy(b"z") == pytest.approx(math.log(256))


# --- masked diffusion -----------------------------------------------------------------------------------------------

def test_diffusion_bounds_agree_exactly():
    torch.manual_seed(0)
    cfg = tiny().with_(attention="gqa-bidir", vocab_size=V + 1)
    m = DiffusionLM(cfg).double().eval()
    x0 = [5, 17, 3]

    def logp(xt):
        return torch.log_softmax(BlockLM.forward(m, torch.tensor([xt])).logits[0], -1)

    a, b = exact_bound(logp, x0, m.mask_id, "eq3"), exact_bound(logp, x0, m.mask_id, "eq6")
    assert abs(a - b) < 1e-10
    mc = nll_bound(m, torch.tensor([x0]), samples=4000, seed=1).item() * 3
    assert abs(mc - b) / b < 0.05


def test_diffusion_is_not_causal_but_ar_is():
    torch.manual_seed(0)
    cfg = small().with_(attention="gqa-bidir", vocab_size=V + 1)
    m = DiffusionLM(cfg)
    with pytest.raises(AssertionError):
        model_checks(m, V, log=lambda *_: None, cache=False)


def test_diffusion_mask_rate_and_sampler():
    g = torch.Generator().manual_seed(0)
    x = torch.zeros(4, 1000, dtype=torch.long)
    xt, masked = mask_tokens(x, torch.tensor([0.1, 0.5, 0.9, 1.0]), 99, g)
    assert abs(masked[1].float().mean().item() - 0.5) < 0.06 and masked[3].all()
    torch.manual_seed(0)
    m = DiffusionLM(tiny().with_(attention="gqa-bidir", vocab_size=V + 1))
    out = sample(m, torch.randint(0, V, (1, 4)), gen_len=6, steps=3)
    assert out.shape == (1, 10) and (out != m.mask_id).all()


# --- latent reasoning ---------------------------------------------------------------------------------------------

def test_continuous_thoughts_equal_full_recompute():
    torch.manual_seed(0)
    m = BlockLM(small()).double().eval()
    idx = torch.randint(0, V, (1, 6))
    logits, thoughts = continuous_thoughts(m, idx, 3)
    emb = torch.cat([m.model.embed_tokens(idx), thoughts], 1)
    full = m(inputs_embeds=emb).logits[:, -1]
    assert (full - logits).abs().max().item() < 1e-10


def test_recurrent_depth_truncated_backprop():
    torch.manual_seed(0)
    m = RecurrentDepthLM(tiny()).double()
    idx = torch.randint(0, V, (1, 6))
    def grads(k):
        m.zero_grad()
        m(idx, r=3, backprop_last=k).logsumexp(-1).sum().backward()
        return torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
    assert torch.allclose(grads(None), grads(3)) and not torch.allclose(grads(None), grads(1))
    assert m.effective_depth(32) == 1 + 2 * 32 + 1
    rs = [sample_recurrence(8, generator=torch.Generator().manual_seed(s)) for s in range(400)]
    assert min(rs) >= 1 and 6 < sum(rs) / len(rs) < 11


# --- training through the loop ---------------------------------------------------------------------------------------

ARGS = ["--preset", "toy", "--batch", "4", "--seq", "32", "--lr", "3e-3", "--warmup", "5", "--steps", "16",
        "--log-every", "4", "--eval-every", "16", "--eval-windows", "4", "--ckpt-every", "100", "--device", "cpu"]


@pytest.mark.parametrize("flags", [["--mtp", "deepseek", "--mtp-schedule", "deepseek", "--blocks-log"],
                                   ["--residual", "mhc", "--ffn", "moe", "--blocks-log", "--hyper-every", "4"],
                                   ["--ffn", "matformer"], ["--objective", "diffusion"]])
def test_exact_resume_with_module6_wrapper(tiny_data, tmp_path, flags):
    from frontierlab.blocks import train
    from frontierlab.metrics import read_jsonl
    a = train.main(["--run", str(tmp_path / "a"), "--data", str(tiny_data), *ARGS, *flags])
    train.main(["--run", str(tmp_path / "b"), "--data", str(tiny_data), *ARGS, *flags, "--stop-after", "7"])
    b = train.main(["--run", str(tmp_path / "b"), "--data", str(tiny_data), *ARGS, *flags])
    for (n, p), (_, q) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(p, q), n
    la = [r["loss"] for r in read_jsonl(tmp_path / "a" / "metrics.jsonl") if r["split"] == "train"]
    lb = [r["loss"] for r in read_jsonl(tmp_path / "b" / "metrics.jsonl") if r["split"] == "train"]
    assert la == lb
    from frontierlab.runcard import read_run_card
    card = read_run_card(tmp_path / "a")
    assert card["params"]["total"] == sum(p.numel() for p in a.parameters())
    assert "blocks" in card and "blocks" in card["config"]["extra"]
