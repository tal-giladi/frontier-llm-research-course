import math

import pytest
import torch
import torch.nn.functional as F

from frontierlab.model import LM, baseline0, toy
from frontierlab.perf import dist as pdist
from frontierlab.perf.chunked_ce import (chunked_cross_entropy, hidden_states, lm_loss_chunked,
                                         reference_cross_entropy)
from frontierlab.perf.memory import SavedTensors, logits_bytes, saved_activation_bytes
from frontierlab.perf.profiling import busy_fraction, profile_steps, summarize
from frontierlab.perf.roofline import (HARDWARE, decode_attention, gemm, predicted_step_time, rmsnorm,
                                       roofline_time, softmax, step_time_model)
from frontierlab.perf.timing import Timing, benchmark, interleaved, speedup


# ---- roofline -------------------------------------------------------------------------------

def test_ridge_points_and_datasheet_peaks():
    h100 = HARDWARE["H100-SXM"]
    assert h100.peak_flops == 989e12 and h100.mem_bw == 3.35e12
    assert abs(h100.ridge - 295.2) < 0.1
    assert abs(HARDWARE["A100-SXM-80GB"].ridge - 153.0) < 0.1


def test_worked_intensities():
    assert gemm(4, 4, 4).flops == 128 and gemm(4, 4, 4).bytes == 96
    assert abs(gemm(8192, 2816, 768).intensity - 562.0) < 0.1          # compute-bound on H100
    assert gemm(1, 2816, 768).intensity < 1.01                         # same weights at decode: memory-bound
    assert rmsnorm(10, 768).intensity == 1.0
    assert softmax(10, 1024).intensity == 1.25
    for S in (1024, 32768):                                            # independent of context and batch
        assert decode_attention(4, 12, 4, S, 64).intensity == 3.0


def test_roofline_time_takes_the_binding_limit():
    hw = HARDWARE["H100-SXM"]
    assert roofline_time(989e12, 1.0, hw) == pytest.approx(1.0)
    assert roofline_time(1.0, 3.35e12, hw) == pytest.approx(1.0)


def test_step_model_is_a_lower_bound_of_the_mfu_prediction():
    cfg, hw = baseline0(), HARDWARE["H100-SXM"]
    m = step_time_model(cfg, 32, 1024, hw, grad_accum=8)
    t100 = predicted_step_time(cfg, 32, 1024, hw, mfu=1.0, grad_accum=8)
    assert m["t_compute_bound"] <= t100 * 1.01                         # GEMMs + attention FLOPs only
    assert m["t_total"] > t100                                         # memory-bound ops add time
    assert predicted_step_time(cfg, 32, 1024, hw, 0.3, 8) == pytest.approx(t100 / 0.3)


# ---- timing ---------------------------------------------------------------------------------

def test_benchmark_counts_and_ci():
    calls = []
    t = benchmark(lambda: calls.append(1), warmup=2, repeats=7)
    assert len(calls) == 9 and len(t.samples) == 7 and len(t.warmup) == 2
    lo, hi = t.ci()
    assert lo <= t.median <= hi


def test_speedup_paired_and_unpaired():
    base = Timing(samples=[2.0, 2.2, 1.9, 2.1, 2.0, 2.05])
    new = Timing(samples=[1.0, 1.1, 0.95, 1.05, 1.0, 1.02])
    s = speedup(base, new)
    assert 1.9 < s["speedup"] < 2.1 and s["ci"][0] > 1.5
    s2 = speedup(base, Timing(samples=[1.0, 1.0, 1.0]))
    assert s2["speedup"] == pytest.approx(2.025)


def test_interleaved_runs_rounds():
    order = []
    res = interleaved({"a": lambda: order.append("a"), "b": lambda: order.append("b")}, warmup=1, rounds=3)
    assert order == ["a", "b"] * 4 and len(res["a"].samples) == 3


# ---- profiling ------------------------------------------------------------------------------

def test_busy_fraction_merges_overlaps():
    assert busy_fraction([(0, 2), (1, 3), (5, 6)], (0, 10)) == pytest.approx(0.4)
    assert busy_fraction([], (0, 1)) == 0.0


def test_summarize_cpu_profile():
    lin = torch.nn.Linear(64, 64)
    x = torch.randn(32, 64)
    prof = profile_steps(lambda: lin(x).sum().backward(), steps=2, warmup=1)
    s = summarize(prof, top=5)
    assert s["steps"] == 2 and s["step_ms"] > 0 and s["n_ops"] > 0
    assert any("mm" in name for name, *_ in s["top"])
    assert 0 < sum(share for *_, share in s["top"]) <= 1.0 + 1e-9


# ---- memory and chunked cross-entropy -------------------------------------------------------

def test_saved_tensors_dedups_and_excludes_params():
    w = torch.nn.Parameter(torch.randn(8, 8))
    x = torch.randn(4, 8, requires_grad=True)
    with SavedTensors(exclude=[w]) as st:
        y = (x @ w).sin()
    assert st.total_bytes == 2 * 4 * 8 * 4                             # x (saved by mm) and x@w (by sin)
    del y


def test_chunked_ce_matches_reference_in_float64():
    torch.manual_seed(0)
    N, C, V = 37, 16, 50
    h = torch.randn(N, C, dtype=torch.float64, requires_grad=True)
    w = torch.randn(V, C, dtype=torch.float64, requires_grad=True)
    t = torch.randint(0, V, (N,))
    t[3] = -100                                                        # ignored position
    ref = reference_cross_entropy(h, w, t)
    gh_ref, gw_ref = torch.autograd.grad(ref, (h, w))
    for chunk in (1, 8, 37, 100):
        out = chunked_cross_entropy(h, w, t, chunk_size=chunk)
        gh, gw = torch.autograd.grad(out, (h, w))
        assert abs(out.item() - ref.item()) < 1e-12
        assert (gh - gh_ref).abs().max() < 1e-12 and (gw - gw_ref).abs().max() < 1e-12


def test_chunked_ce_scales_with_upstream_gradient():
    torch.manual_seed(1)
    h = torch.randn(10, 4, dtype=torch.float64, requires_grad=True)
    w = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    t = torch.randint(0, 7, (10,))
    (3.0 * chunked_cross_entropy(h, w, t, 3)).backward()
    gh = h.grad.clone()
    h.grad = None
    (3.0 * F.cross_entropy(h @ w.t(), t)).backward()
    assert (gh - h.grad).abs().max() < 1e-12


def test_lm_loss_chunked_equals_model_loss_and_grads():
    torch.manual_seed(0)
    cfg = toy(vocab_size=97)
    x = torch.randint(0, 97, (3, 20))
    # float64: against the same hidden states with the ordinary loss (exact up to rounding)
    m64 = LM(cfg).double()
    h = hidden_states(m64, x)[:, :-1]
    ref = reference_cross_entropy(h, m64.lm_head.weight, x[:, 1:])
    g_ref = torch.autograd.grad(ref, list(m64.parameters()))
    out = lm_loss_chunked(m64, x, chunk_size=16)
    g = torch.autograd.grad(out, list(m64.parameters()))
    assert abs(out.item() - ref.item()) < 1e-12
    assert max((a - b).abs().max().item() for a, b in zip(g, g_ref)) < 1e-12
    # float32: against the model's own loss (LM.forward), within fp32 rounding
    m = LM(cfg)
    ref32 = m(x, labels=x).loss
    g32 = torch.autograd.grad(ref32, list(m.parameters()))
    out32 = lm_loss_chunked(m, x, chunk_size=16)
    gc = torch.autograd.grad(out32, list(m.parameters()))
    assert abs(out32.item() - ref32.item()) < 1e-5
    assert max((a - b).abs().max().item() for a, b in zip(gc, g32)) < 1e-6


def test_chunked_ce_removes_the_logits_from_saved_memory():
    torch.manual_seed(0)
    cfg = toy(vocab_size=4096)
    m = LM(cfg)
    x = torch.randint(0, 4096, (4, 64))
    full = saved_activation_bytes(m, lambda: m(x, labels=x).loss).total_bytes
    chunk = saved_activation_bytes(m, lambda: lm_loss_chunked(m, x, 64)).total_bytes
    logits = logits_bytes(4, 63, 4096)
    # gone: the fp32 log-softmax output (B(T-1) x V); added: the stored grad_W (V x C fp32). The head's
    # saved input h and the new grad_h are the same size, so they cancel to within a few KB.
    assert full - chunk == pytest.approx(logits - 4096 * cfg.hidden_size * 4, rel=0.01)


# ---- distributed ------------------------------------------------------------------------------

def test_comm_arithmetic():
    assert pdist.allreduce_bytes_per_rank(100, 4) == 150
    assert pdist.bus_bandwidth(100, 1.0, 4) == 150
    assert pdist.bus_bandwidth(100, 1.0, 4, "all_gather") == 75
    mb = 2**20
    assert pdist.ddp_buckets([10 * mb] * 6, 25 * mb, 1 * mb) == [10 * mb, 30 * mb, 20 * mb]
    assert sum(pdist.ddp_buckets([3, 5, 7], 100)) == 15


def test_gloo_ddp_and_fsdp2_two_ranks():
    for mode in ("ddp", "fsdp2"):
        res = pdist.spawn(pdist.train_step_worker, 2, mode=mode, vocab=257, batch=2, seq=16, steps=2, warmup=1)
        assert len(res) == 2 and all(len(r["sync"]) == 2 and len(r["nosync"]) == 2 for r in res)
        assert all(math.isfinite(t) for r in res for t in r["sync"] + r["nosync"])
