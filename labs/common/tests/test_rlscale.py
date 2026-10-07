"""Tests for frontierlab.rlscale (Module 14): objectives against their derivations, the loop wrapper, the
async model, mismatch and batch invariance, the ScaleRL curve fit, pass@k curves."""

import math

import numpy as np
import pytest
import torch

from frontierlab.rlscale import asyncsim, curves, mismatch, passk, stability
from frontierlab.rlscale import objectives as O


def batch(seed=0, B=4, R=5, spread=0.3):
    g = torch.Generator().manual_seed(seed)
    old = torch.log(torch.rand(B, R, generator=g, dtype=torch.float64) * 0.8 + 0.1)
    logp = old + spread * torch.randn(B, R, generator=g, dtype=torch.float64)
    lengths = torch.tensor([5, 2, 3, 1][:B])
    mask = (torch.arange(R)[None] < lengths[:, None]).double()
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5][:B], dtype=torch.float64)
    return logp, old, adv, mask


# --------------------------------------------------------------------------- closed-form gradients

def test_grpo_gradient_matches_derivation():
    logp, old, adv, mask = batch()
    w = O.gradient_weights("grpo", logp, old, adv, mask)
    r = torch.exp(logp - old)
    live = ~(((adv[:, None] > 0) & (r > 1.2)) | ((adv[:, None] < 0) & (r < 0.8)))
    want = r * adv[:, None] / mask.sum(-1, keepdim=True) / 4 * live * mask
    assert torch.allclose(w, want, atol=1e-12)


def test_dapo_and_drgrpo_gradients():
    logp, old, adv, mask = batch(1)
    r = torch.exp(logp - old)
    live = ~(((adv[:, None] > 0) & (r > 1.28)) | ((adv[:, None] < 0) & (r < 0.8)))
    want = r * adv[:, None] / mask.sum() * live * mask
    assert torch.allclose(O.gradient_weights("dapo", logp, old, adv, mask), want, atol=1e-12)
    live2 = ~(((adv[:, None] > 0) & (r > 1.2)) | ((adv[:, None] < 0) & (r < 0.8)))
    want2 = r * adv[:, None] / (4 * 7) * live2 * mask
    assert torch.allclose(O.gradient_weights("drgrpo", logp, old, adv, mask, norm_len=7), want2, atol=1e-12)


def test_gspo_gradient_is_shared_per_sequence():
    logp, old, adv, mask = batch(2, spread=1e-4)              # inside the tiny clip range
    s = torch.exp(((logp - old) * mask).sum(-1) / mask.sum(-1))
    want = (s * adv / mask.sum(-1) / 4)[:, None] * mask
    assert torch.allclose(O.gradient_weights("gspo", logp, old, adv, mask), want, atol=1e-12)
    # far outside the range: sequences moved in the advantage's direction lose all their gradient
    logp2 = old + 0.5 * mask * torch.sign(adv)[:, None]
    w = O.gradient_weights("gspo", logp2, old, adv, mask)
    assert torch.count_nonzero(w) == 0


def test_cispo_keeps_gradient_where_ppo_clips():
    logp, old, adv, mask = batch(3, spread=0.8)
    r = torch.exp(logp - old)
    want = torch.clamp(r, max=1.28) * adv[:, None] / mask.sum() * mask
    assert torch.allclose(O.gradient_weights("cispo", logp, old, adv, mask), want, atol=1e-12)
    clipped = ((adv[:, None] > 0) & (r > 1.28)) & mask.bool()
    assert clipped.any()
    assert (O.gradient_weights("dapo", logp, old, adv, mask)[clipped] == 0).all()
    assert (O.gradient_weights("cispo", logp, old, adv, mask)[clipped] != 0).all()


def test_on_policy_equivalences():
    logp, _, adv, mask = batch(4)
    old = logp.clone()
    g = {n: O.gradient_weights(n, logp, old, adv, mask) for n in O.OBJECTIVES}
    assert torch.allclose(g["grpo"], g["gspo"], atol=1e-12)        # same per-sequence mean at ratio 1
    assert torch.allclose(g["dapo"], g["cispo"], atol=1e-12)       # same token mean at ratio 1
    assert torch.allclose(g["drgrpo"] * 4 * logp.shape[1] / mask.sum(), g["dapo"], atol=1e-12)


def test_gspo_token_equals_gspo_with_sequence_advantages():
    logp, old, adv, mask = batch(5, spread=2e-4)
    a = O.gradient_weights("gspo", logp, old, adv, mask)
    b = O.gradient_weights("gspo", logp, old, adv, mask, loss_fn=None)
    lp = logp.clone().requires_grad_(True)
    loss, _ = O.gspo_token_loss(lp, old, adv, mask)
    loss.backward()
    assert torch.allclose(-lp.grad, a, atol=1e-12) and torch.allclose(a, b)
    v1, _ = O.gspo_loss(logp, old, adv, mask)
    assert torch.allclose(v1, O.gspo_token_loss(logp, old, adv, mask)[0], atol=1e-12)


def test_sequence_ratio_is_length_normalised():
    old = torch.zeros(1, 4, dtype=torch.float64)
    lp = torch.log(torch.tensor([[2.0, 0.5, 2.0, 0.5]], dtype=torch.float64))
    m2, m4 = torch.tensor([[1.0, 1, 0, 0]]).double(), torch.ones(1, 4).double()
    assert math.isclose(float(torch.exp(O.sequence_log_ratio(lp, old, m2))), 1.0, abs_tol=1e-12)
    lp3 = torch.log(torch.tensor([[2.0, 2.0, 2.0, 2.0]], dtype=torch.float64))
    assert math.isclose(float(torch.exp(O.sequence_log_ratio(lp3, old, m2))), 2.0, rel_tol=1e-12)
    assert math.isclose(float(torch.exp(O.sequence_log_ratio(lp3, old, m4))), 2.0, rel_tol=1e-12)


@pytest.mark.parametrize("name", list(O.OBJECTIVES))
def test_padding_cannot_change_the_loss(name):
    logp, old, adv, mask = batch(6)
    noisy = logp + (1 - mask) * 5.0
    kw = {"norm_len": 5} if name == "drgrpo" else {}
    a = O.gradient_weights(name, logp, old, adv, mask, **kw)
    b = O.gradient_weights(name, noisy, old, adv, mask, **kw)
    assert torch.allclose(a, b, atol=1e-12)
    assert (b[mask == 0] == 0).all()


def test_hook_and_truncated_is_weight():
    logp, old, adv, mask = batch(7)
    w = torch.full_like(logp, 0.5)
    hook = O.as_hook("dapo")
    l1, _ = hook(logp, old, adv, mask, aggregation="ignored", group_size=2, eps_low=9, eps_high=9, ratio="none",
                 norm_len=5)
    l2, _ = hook(logp, old, adv, mask, is_weight=w, norm_len=5)
    assert torch.allclose(l2, 0.5 * l1)
    assert torch.allclose(l1, O.dapo_loss(logp, old, adv, mask)[0])


# --------------------------------------------------------------------------- loop wrapper and controls

def test_objective_hooks_run_in_the_course_loop(tmp_path):
    from frontierlab.posttrain.rl import RLConfig
    from frontierlab.metrics.jsonl import read_jsonl
    from frontierlab.rlscale import runner
    init = runner.initial_policy(tmp_path / "init.pt")
    base = RLConfig(init=str(init), steps=2, prompts=2, group=4, eval_every=2, eval_n=8, ckpt_every=2)
    for name in O.OBJECTIVES:
        cfg = runner.objective_config(base, name, run=str(tmp_path / name))
        runner.train_objective(cfg, name)
        rows = read_jsonl(tmp_path / name / "metrics.jsonl")
        assert [r["step"] for r in rows if r["split"] == "train"] == [1, 2]
    cfg = runner.objective_config(base, "grpo", run=str(tmp_path / "ctl"))
    runner.train_objective(cfg, "grpo", control="random")
    rows = read_jsonl(tmp_path / "ctl" / "metrics.jsonl")
    ev = [r for r in rows if r["split"] == "eval"]
    assert ev and "greedy_acc" in ev[0]
    from frontierlab.posttrain import rl
    assert rl.evaluate.__name__ == "evaluate" and rl.Sampler.__name__ == "Sampler"   # patches undone


def test_bounded_staleness_sampler_respects_the_bound(tmp_path):
    from frontierlab.posttrain.rl import RLConfig
    from frontierlab.rlscale import runner
    init = runner.initial_policy(tmp_path / "init.pt")
    cfg = runner.objective_config(RLConfig(init=str(init), steps=6, prompts=2, group=4, eval_every=6, eval_n=8,
                                           staleness=2), "grpo", run=str(tmp_path / "async"))
    out = runner.train_objective(cfg, "grpo", lags=[0, 3, 1, 2, 5, 2])
    assert out["lags_used"] == [0, 1, 1, 2, 2, 2]      # min(want, bound, snapshots available - 1)


def test_control_rewards():
    from frontierlab.rlscale.runner import format_verify, random_verify
    assert format_verify("42", True, None) and not format_verify("42", False, None)
    assert not format_verify("#42#", True, None)
    torch.manual_seed(0)
    draws = [random_verify("x", True, None) for _ in range(4000)]
    assert abs(np.mean(draws) - 0.5) < 0.03


def test_stability_metrics():
    rows = [{"split": "train", "step": i, "entropy": 1.0 - i / 100, "grad_norm": 1.0 if i != 50 else 20.0,
             "ratio_max": 1.1, "clip_frac": 0.01, "kl_exact": 0.1} for i in range(100)]
    rows += [{"split": "eval", "step": s, "sampled_acc": a} for s, a in ((0, .2), (25, .4), (50, .25), (100, .3))]
    st = stability.stability(rows)
    assert st["grad_spikes"] == 1 and st["collapsed"] and math.isclose(st["eval_drawdown"], 0.15)
    assert math.isclose(st["entropy_drop"], 0.9, abs_tol=1e-9)


# --------------------------------------------------------------------------- async model and mismatch

def test_async_schedule_by_hand():
    sync = asyncsim.simulate([2, 2, 2], 1.0, 0)
    assert sync["total_s"] == 9.0 and sync["lags"] == [0, 0, 0]
    one = asyncsim.simulate([2, 2, 2], 1.0, 1)
    assert one["total_s"] == 7.0 and one["lags"] == [0, 1, 1]
    rng = np.random.default_rng(0)
    g = [asyncsim.batch_gen_time(L, 0.01) for L in asyncsim.lognormal_lengths(rng, 50, 64, 300, 0.8, 4096)]
    for k in (0, 1, 2, 4, 8):
        r = asyncsim.simulate(g, 2.0, k)
        assert r["max_lag"] <= k
    rows = asyncsim.speedup_table(g, 2.0)
    assert rows[0]["speedup"] == 1.0 and all(a["speedup"] <= b["speedup"] + 1e-9 for a, b in zip(rows, rows[1:]))


def test_mismatch_stats_and_fixed_tile_invariance():
    t = torch.log(torch.tensor([[0.5, 0.2, 0.9]]))
    s = torch.log(torch.tensor([[0.25, 0.2, 0.9]]))
    st = mismatch.mismatch_stats(t, s, torch.ones(1, 3))
    assert math.isclose(st["max_ratio"], 2.0, rel_tol=1e-6) and st["k3"] > 0
    torch.manual_seed(0)
    W, x = torch.randn(256, 256), torch.randn(100, 256)
    dep = mismatch.batch_dependence(lambda a: mismatch.fixed_tile_matmul(a, W), x)
    assert all(v == 0.0 for v in dep.values())
    assert torch.allclose(mismatch.fixed_tile_matmul(x, W), x @ W, atol=1e-4)


# --------------------------------------------------------------------------- curves and pass@k

def test_sigmoid_fit_recovers_parameters_on_a_long_curve():
    C = np.geomspace(1.5e3, 1e5, 30)
    y = curves.sigmoid_curve(C, 0.645, 1.70, 8e3, 0.30)
    fit = curves.fit_sigmoid(C, y, 0.30)
    assert abs(fit["A"] - 0.645) < 1e-3 and abs(fit["B"] - 1.70) < 1e-2 and abs(fit["C_mid"] / 8e3 - 1) < 1e-2


def test_asymptote_unidentified_on_a_short_run():
    rng = np.random.default_rng(0)
    C_long = np.geomspace(1.5e3, 1e5, 30)
    C_short = np.geomspace(1.5e3, 4e3, 12)
    yl = curves.sigmoid_curve(C_long, 0.645, 1.70, 8e3, 0.30) + rng.normal(0, 0.005, C_long.size)
    ys = curves.sigmoid_curve(C_short, 0.645, 1.70, 8e3, 0.30) + rng.normal(0, 0.005, C_short.size)
    long_ = curves.profile_asymptote(C_long, yl, 0.30)
    short = curves.profile_asymptote(C_short, ys, 0.30)
    lo, hi = long_["interval"]
    assert long_["bounded_above"] and lo <= 0.645 <= hi and hi - lo < 0.05
    assert not short["bounded_above"]


def test_passk_curve_and_crossover():
    n, ks = 16, [1, 2, 4, 8, 16]
    rl = [8, 8, 8, 0, 0]          # sharpened: right half the time on 3 problems, never on 2
    base = [3, 3, 3, 1, 1]        # spread out: sometimes right everywhere
    cr, cb = passk.curve(rl, n, ks), passk.curve(base, n, ks)
    assert cr[0] > cb[0] and cb[-1] > cr[-1]
    assert passk.crossover(cr, cb, ks) in ks[1:]
    assert passk.solved_sets(rl, base) == {"both": 3, "only_a": 0, "only_b": 2, "neither": 0}
    d = passk.paired_curve_diff(rl, base, n, ks, n_boot=200)
    assert d[0]["diff"] > 0 and d[-1]["diff"] < 0
