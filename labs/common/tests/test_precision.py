"""Tests of frontierlab.precision (Module 8): exact formats, scaling, recipes, the training wrapper."""

import math

import pytest
import torch
import torch.nn.functional as F

from frontierlab.metrics import read_jsonl
from frontierlab.precision import accum, scaling_law
from frontierlab.precision.formats import (E2M1, E4M3, E5M2, INT4, decode, encode, representable_values, round_pow2,
                                           round_to_format)
from frontierlab.precision.hadamard import apply_rht, hadamard, rht_matrix
from frontierlab.precision.linear import QLinear, Recipe, get_recipe, parse_keep, ptq_, select_linears, swap_linears
from frontierlab.precision.qat import attach_qat, export_, fake_quant
from frontierlab.precision.quant import QuantSpec, block_scales, qdq, quant_error, quantize, _blocks

TORCH_FP8 = ((E4M3, torch.float8_e4m3fn), (E5M2, torch.float8_e5m2))


def wide_values(n=200_000, lo=-20, hi=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g) * torch.exp2(torch.randint(lo, hi, (n,), generator=g).float())


# ------------------------------------------------------------------------------------------- formats

@pytest.mark.parametrize("fmt,dt", TORCH_FP8)
def test_decode_every_bit_pattern_matches_torch(fmt, dt):
    codes = torch.arange(256, dtype=torch.uint8)
    ref, mine = codes.view(dt).double(), decode(codes, fmt)
    same = (ref == mine) | (ref.isnan() & mine.isnan())
    assert bool(same.all())


@pytest.mark.parametrize("fmt,dt", TORCH_FP8)
def test_round_and_encode_match_torch_casts_in_range(fmt, dt):
    x = wide_values().clamp(-fmt.max_normal, fmt.max_normal)       # includes subnormals and exact ties
    r = round_to_format(x, fmt)
    assert torch.equal(r, x.to(dt).float())
    assert torch.equal(encode(r, fmt), r.to(dt).view(torch.uint8).long())
    ties = torch.tensor([1.0625, 1.1875, 2.0 ** -9 * 1.5]) if fmt is E4M3 else torch.tensor([1.125, 1.375])
    assert torch.equal(round_to_format(ties, fmt), ties.to(dt).float())       # round half to even


def test_overflow_rules():
    big = torch.tensor([470.0, 1e6, -1e6])
    assert round_to_format(big, E4M3).tolist() == [448.0, 448.0, -448.0]                 # saturate (recipes)
    assert torch.isnan(round_to_format(big, E4M3, saturate=False)).all()                # OFP8 OVF: NaN for E4M3
    y = torch.tensor([57343.0, 61439.0, 61440.0, 1e6])
    assert torch.equal(round_to_format(y, E5M2, saturate=False), y.to(torch.float8_e5m2).float())   # IEEE: inf
    assert round_to_format(y, E5M2).tolist() == [57344.0] * 4


def test_format_tables():
    assert representable_values(E2M1).tolist() == [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    assert (E4M3.max_normal, E4M3.min_normal, E4M3.min_subnormal) == (448.0, 2 ** -6, 2 ** -9)
    assert (E5M2.max_normal, E5M2.min_normal, E5M2.min_subnormal) == (57344.0, 2 ** -14, 2 ** -16)
    assert E4M3.emax == 8 and E2M1.emax == 2
    assert len(representable_values(E4M3)) == 127                  # 0 and 126 positive finite values
    assert round_to_format(torch.tensor([7.6, -9.0, 2.5]), INT4).tolist() == [7.0, -7.0, 2.0]


def test_e8m0_rounding_matches_torch():
    x = torch.rand(10_000) * 100 + 1e-3
    assert torch.equal(round_pow2(x, "nearest"), x.to(torch.float8_e8m0fnu).float())
    assert bool((round_pow2(x, "ceil") >= x).all()) and bool((round_pow2(x, "floor") <= x).all())


def test_stochastic_rounding_is_unbiased_and_lands_on_neighbours():
    x = torch.full((200_000,), 1.3)                                 # E2M1 neighbours 1.0 and 1.5
    g = torch.Generator().manual_seed(0)
    r = round_to_format(x, E2M1, "sr", g)
    assert set(torch.unique(r).tolist()) == {1.0, 1.5}
    assert abs(r.mean().item() - 1.3) < 0.005                        # E[SR(x)] = x; RNE would give 1.5
    assert round_to_format(torch.tensor([1.3]), E2M1).item() == 1.5


# ------------------------------------------------------------------------------------------- scaling

def test_tensorwise_maps_amax_to_format_max():
    x = torch.randn(64, 64)
    q, s = quantize(x, QuantSpec("e4m3", (0, 0), "fp32"))
    assert q.abs().max().item() == 448.0 and s.numel() == 1


def test_block_scales_shapes_and_tile_equals_row_when_width_matches():
    x = torch.randn(10, 128)
    assert torch.equal(qdq(x, QuantSpec("e4m3", (1, 128))), qdq(x, QuantSpec("e4m3", (1, 0))))
    xb, _ = _blocks(torch.randn(300, 200), 128, 128)                # padded to 384 x 256
    assert block_scales(xb, QuantSpec("e4m3", (128, 128))).shape == (3, 1, 2, 1)


@pytest.mark.parametrize("block", [(16, 16), (128, 128), (0, 0)])
def test_square_blocks_quantise_w_and_wt_identically(block):
    w = torch.randn(96, 160)
    spec = QuantSpec("e2m1" if block == (16, 16) else "e4m3", block, "nvfp4" if block == (16, 16) else "fp32")
    assert torch.equal(qdq(w.t(), spec).t(), qdq(w, spec))


def test_one_d_blocks_quantise_w_and_wt_differently():
    w = torch.randn(64, 64)
    spec = QuantSpec("e2m1", (1, 16), "nvfp4")
    assert not torch.equal(qdq(w.t(), spec).t(), qdq(w, spec))


def test_bits_per_value():
    assert QuantSpec("e2m1", (1, 32), "e8m0").bits_per_value() == 4.25          # gpt-oss MXFP4
    assert QuantSpec("e2m1", (1, 16), "nvfp4").bits_per_value() == 4.5
    assert QuantSpec("e4m3", (1, 128), "fp32").bits_per_value() == 8.25


def test_scale_kinds():
    x = torch.randn(8, 64) * 3
    _, s = quantize(x, QuantSpec("e2m1", (1, 32), "e8m0"))
    assert torch.equal(s, torch.exp2(torch.log2(s).round()))                    # powers of two
    sat = quant_error(x, QuantSpec("e2m1", (1, 32), "e8m0"))["saturated"]
    assert sat > 0                                                              # OCP rule clamps the top binade
    assert quant_error(x, QuantSpec("e2m1", (1, 32), "pow2"))["saturated"] == 0
    # NVFP4: tensor decode scale amax / (6 * 448); block scales are E4M3 values times it
    xb, _ = _blocks(x, 1, 16)
    s = block_scales(xb, QuantSpec("e2m1", (1, 16), "nvfp4"))
    t = x.abs().max() / (6 * 448)
    assert torch.equal(round_to_format(s / t, E4M3), (s / t).float()) or torch.allclose(s / t, round_to_format(s / t, E4M3), rtol=1e-6)


def test_fine_grained_scaling_beats_tensorwise_with_outliers():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, 512, generator=g) * 1e-3
    x[3, 7] = 1e4                                                                # one activation outlier
    tw = quant_error(x, QuantSpec("e4m3", (0, 0)))
    tile = quant_error(x, QuantSpec("e4m3", (1, 128)))
    assert tw["underflow"] > 0.1 and tile["underflow"] < 1e-3
    assert tile["rel_err"] < tw["rel_err"]


# ------------------------------------------------------------------------------------------- hadamard

def test_hadamard_orthogonal_and_rht_preserves_wgrad():
    for d in (2, 16, 32):
        H = hadamard(d)
        assert torch.allclose(H @ H.t(), torch.eye(d, dtype=torch.float64), atol=1e-12)
    Q = rht_matrix(16, seed=3)
    assert torch.allclose(Q @ Q.t(), torch.eye(16, dtype=torch.float64), atol=1e-12)
    g, x = torch.randn(24, 64, dtype=torch.float64), torch.randn(40, 64, dtype=torch.float64)   # (N, M), (K, M)
    exact = g @ x.t()
    assert torch.allclose(apply_rht(g, 16, 3) @ apply_rht(x, 16, 3).t(), exact, atol=1e-10)


def test_rht_spreads_token_outliers_so_fewer_values_underflow():
    """A 30x outlier every 64 tokens makes FP4 flush most of its 16-block to zero; after the RHT the block is a
    mixture and far fewer values underflow. (The Frobenius error of the whole GEMM is NOT smaller at this size —
    lesson 08.3 measures that; the paper's evidence for RHT is training loss at scale.)"""
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(48, 1024, generator=gen, dtype=torch.float64)
    x[:, ::64] *= 30
    spec = QuantSpec("e2m1", (1, 16), "nvfp4")
    assert quant_error(apply_rht(x, 16, 0), spec)["underflow"] < 0.5 * quant_error(x, spec)["underflow"]


# ------------------------------------------------------------------------------------------- QLinear

def make_q(recipe, n_in=64, n_out=48, seed=0):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(n_in, n_out, bias=False).double()
    return lin, QLinear.from_linear(lin, recipe)


def test_qlinear_without_quantisers_is_exact():
    lin, q = make_q(Recipe("identity"))
    x = torch.randn(5, 7, 64, dtype=torch.float64, requires_grad=True)
    y1, y2 = lin(x), q(x)
    assert torch.equal(y1, y2)
    g = torch.randn_like(y1)
    gx1, gw1 = torch.autograd.grad(y1, (x, lin.weight), g)
    gx2, gw2 = torch.autograd.grad(y2, (x, q.weight), g)
    assert torch.allclose(gx1, gx2, atol=1e-12) and torch.allclose(gw1, gw2, atol=1e-12)
    assert q.weight is lin.weight


def test_qlinear_gemms_follow_the_recipe():
    r = get_recipe("fp8-deepseek")
    lin, q = make_q(r, 256, 128)
    x = torch.randn(32, 256, dtype=torch.float64, requires_grad=True)
    y = q(x)
    w = lin.weight
    assert torch.allclose(y, qdq(x, r.act) @ qdq(w, r.weight).t(), atol=1e-12)
    g = torch.randn_like(y)
    gx, gw = torch.autograd.grad(y, (x, w), g)
    assert torch.allclose(gx, qdq(g, r.grad) @ qdq(w.t(), r.weight).t(), atol=1e-12)
    assert torch.allclose(gw, qdq(g.t(), r.grad) @ qdq(x.t(), r.act).t(), atol=1e-12)
    # and the emulated gradients are close to the exact ones (FP8: about 1e-2 relative)
    ex, ew = torch.autograd.grad(F.linear(x, w), (x, w), g)
    assert (gx - ex).norm() / ex.norm() < 0.05 and (gw - ew).norm() / ew.norm() < 0.05


def test_nvfp4_backward_uses_rht_and_stochastic_rounding():
    r = get_recipe("nvfp4")
    lin, q = make_q(r, 64, 32)
    x = torch.randn(64, 64, dtype=torch.float64, requires_grad=True)
    g = torch.randn(64, 32, dtype=torch.float64)
    torch.manual_seed(5)
    gw_a = torch.autograd.grad(q(x), q.weight, g)[0]
    torch.manual_seed(6)
    gw_b = torch.autograd.grad(q(x), q.weight, g)[0]
    assert not torch.equal(gw_a, gw_b)                                         # SR on dy draws random numbers
    torch.manual_seed(5)
    assert torch.equal(torch.autograd.grad(q(x), q.weight, g)[0], gw_a)       # ... from the global RNG
    ex = g.t() @ x
    assert (gw_a - ex).norm() / ex.norm() < 0.3                                 # FP4: a coarse but unbiased estimate


def test_sr_gradient_is_unbiased_where_rne_is_not():
    """Averaged over many draws, the SR weight gradient converges to the exact one; RNE keeps a fixed bias."""
    torch.manual_seed(0)
    x = torch.randn(64, 32, dtype=torch.float64)
    g = torch.randn(64, 16, dtype=torch.float64) * 0.3
    exact = g.t() @ x
    sr = Recipe("sr", grad=QuantSpec("e2m1", (1, 16), "nvfp4", "sr"))
    rne = Recipe("rne", grad=QuantSpec("e2m1", (1, 16), "nvfp4", "rne"))
    errs = {}
    for r in (sr, rne):
        lin, q = make_q(r, 32, 16)
        acc = torch.zeros_like(exact)
        for _ in range(300):
            xx = x.clone().requires_grad_(True)
            acc += torch.autograd.grad(q(xx), q.weight, g)[0]
        errs[r.name] = ((acc / 300 - exact).norm() / exact.norm()).item()
    assert errs["sr"] < 0.5 * errs["rne"]


def test_swap_keeps_state_dict_and_selection(tiny_cfg_model):
    model = tiny_cfg_model
    before = {k: v.clone() for k, v in model.state_dict().items()}
    names = swap_linears(model, "nvfp4", keep_blocks=parse_keep("last1", 2))
    assert names and all(".layers.0." in n for n in names) and not any("lm_head" in n for n in names)
    assert len(names) == 7                                                      # q, k, v, o, gate, up, down
    after = model.state_dict()
    assert list(after) == list(before) and all(torch.equal(after[k], before[k]) for k in before)
    assert parse_keep("first2,last8", 12) == {0, 1, 4, 5, 6, 7, 8, 9, 10, 11}
    assert len(select_linears(model, only="mlp")) == 6


@pytest.fixture
def tiny_cfg_model():
    from frontierlab.model import LM, toy
    torch.manual_seed(0)
    return LM(toy(vocab_size=64).with_(num_hidden_layers=2))


def test_qat_export_and_ptq(tiny_cfg_model):
    model = tiny_cfg_model
    x = torch.randint(0, 64, (2, 16))
    ref = model(x).logits.detach()
    import copy
    m2 = copy.deepcopy(model)
    attach_qat(m2, "int4-g32")
    fq = m2(x).logits.detach()
    assert not torch.allclose(fq, ref)
    export_(m2)
    assert torch.allclose(m2(x).logits, fq, atol=1e-5)
    m3 = copy.deepcopy(model)
    info = ptq_(m3, "int4-g32")
    assert torch.allclose(m3(x).logits, fq, atol=1e-5)                         # PTQ == exported fake quant at step 0
    assert info["matrices"] == 14 and abs(info["bits_per_weight"] - (4 + 32 / 32)) < 1e-9
    w = torch.randn(8, 32, requires_grad=True)
    fake_quant(w, "mxfp4").sum().backward()
    assert torch.equal(w.grad, torch.ones_like(w))                              # straight-through


# ------------------------------------------------------------------------------------------- accumulation

def test_limited_accumulation_error_grows_and_promotion_bounds_it():
    e1 = accum.accumulation_error(512, n=32)
    e2 = accum.accumulation_error(4096, n=32)
    ep = accum.accumulation_error(4096, n=32, promote_every=128)
    assert e2["max_rel_err"] > 3 * e1["max_rel_err"]
    assert ep["max_rel_err"] < 0.1 * e2["max_rel_err"]
    a = torch.rand(4, 300, dtype=torch.float64)
    assert torch.allclose(accum.limited_dot(a, a, mant_bits=52, mode="rne"), (a * a).sum(-1), rtol=1e-14)


# ------------------------------------------------------------------------------------------- scaling law

def test_scaling_law_closed_form_matches_grid_and_fit_recovers_parameters():
    for gamma in (2.0, 2.6745, 4.0):
        grid = scaling_law.optimal_precision(1e21, gamma=gamma, P_grid=[p / 20 for p in range(40, 400)])["P_star"]
        assert abs(grid - scaling_law.p_star_closed_form(gamma)) < 0.06
    assert abs(scaling_law.optimal_precision(1e19)["P_star"] - scaling_law.optimal_precision(1e23)["P_star"]) < 1e-9
    N = [1e5, 2e5, 4e5] * 4
    P = [3] * 3 + [4] * 3 + [6] * 3 + [math.inf] * 3
    L = [5.0 * (n * ((1 - math.exp(-p / 2.0)) if p != math.inf else 1.0)) ** -0.3 + 2.0 for n, p in zip(N, P)]
    fit = scaling_law.fit_precision_law(N, P, L, alphas=[0.2, 0.25, 0.3, 0.35], gammas=[1.5, 2.0, 2.5])
    assert fit["gamma"] == 2.0 and fit["alpha"] == 0.3 and fit["rms"] < 1e-9
    pw = scaling_law.fit_power([1, 2, 4, 8], [0.1 * d ** 0.5 for d in (1, 2, 4, 8)])
    assert abs(pw["p"] - 0.5) < 1e-9


# ------------------------------------------------------------------------------------------- the wrapper

ARGS = ["--preset", "toy", "--batch", "4", "--seq", "32", "--lr", "3e-3", "--warmup", "4", "--steps", "12",
        "--log-every", "1", "--eval-every", "100", "--eval-windows", "4", "--ckpt-every", "100", "--device", "cpu"]


@pytest.mark.parametrize("extra", [["--recipe", "nvfp4", "--keep-high", "last1", "--precision-log", "--precision-every", "1"],
                                   ["--recipe", "fp8-deepseek"],
                                   ["--recipe", "int4-qat", "--optimizer", "muon"]])
def test_exact_resume_with_precision_wrapper(tiny_data, tmp_path, extra):
    from frontierlab.precision.train import main
    a = main(["--run", str(tmp_path / "a"), "--data", str(tiny_data), *ARGS, *extra])
    main(["--run", str(tmp_path / "b"), "--data", str(tiny_data), *ARGS, *extra, "--stop-after", "5"])
    b = main(["--run", str(tmp_path / "b"), "--data", str(tiny_data), *ARGS, *extra])
    for (n, p), (_, q) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(p, q), n
    la = [r["loss"] for r in read_jsonl(tmp_path / "a" / "metrics.jsonl") if r["split"] == "train"]
    lb = [r["loss"] for r in read_jsonl(tmp_path / "b" / "metrics.jsonl") if r["split"] == "train"]
    assert la == lb and len(la) == 12
    assert any(isinstance(m, QLinear) for m in a.modules())
    if "--precision-log" in extra:
        rows = read_jsonl(tmp_path / "a" / "precision.jsonl")
        assert [r["step"] for r in rows] == list(range(1, 13))
        assert all(0 < r["grad_rel_err"] < 1 and "worst_grad_layer" in r for r in rows)


def test_wrapper_run_card_and_baseline_is_the_plain_loop(tiny_data, tmp_path):
    from frontierlab.precision.train import main
    from frontierlab.runcard import read_run_card
    from frontierlab.train.loop import main as loop_main
    a = main(["--run", str(tmp_path / "bf16"), "--data", str(tiny_data), *ARGS])
    b = loop_main(["--run", str(tmp_path / "plain"), "--data", str(tiny_data), *ARGS])
    assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))
    main(["--run", str(tmp_path / "fp8"), "--data", str(tiny_data), *ARGS, "--recipe", "fp8-tensorwise", "--steps", "2"])
    card = read_run_card(tmp_path / "fp8")
    assert card["precision"]["recipe"] == "fp8-tensorwise" and card["precision"]["emulated"] is True
    assert card["precision"]["quantized_linears"] == 28
    assert read_run_card(tmp_path / "bf16")["precision"]["emulated"] is False


def test_torchao_refuses_cpu(tiny_data, tmp_path):
    from frontierlab.precision.train import main
    with pytest.raises(SystemExit):
        main(["--run", str(tmp_path / "x"), "--data", str(tiny_data), *ARGS, "--torchao", "tensorwise"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torchao Float8 needs a CUDA GPU (main path, pilot)")
def test_torchao_float8_converts_on_gpu():
    pytest.importorskip("torchao")
    from frontierlab.model import LM, toy
    from frontierlab.precision.torchao_path import convert_torchao_float8, fp8_capable
    if not fp8_capable():
        pytest.skip("GPU without FP8 tensor cores")
    m = LM(toy(vocab_size=256)).cuda()
    names = convert_torchao_float8(m, "rowwise")
    assert len(names) == 28
    with torch.autocast("cuda", torch.bfloat16):
        m(torch.randint(0, 256, (2, 64), device="cuda"), labels=torch.randint(0, 256, (2, 64), device="cuda")).loss.backward()


def test_projected_speedup_is_bounded_by_amdahl():
    from frontierlab.model import PRESETS
    from frontierlab.precision.cost import projected_speedup
    for preset in ("pilot-30m", "baseline0"):
        cfg = PRESETS[preset](vocab_size=32768)
        for hw, lowp in (("H100-SXM", "fp8"), ("L4", "fp8"), ("B200", "fp4")):
            fused = projected_speedup(cfg, 16, 1024, hw, lowp, fused_casts=True)
            unfused = projected_speedup(cfg, 16, 1024, hw, lowp, fused_casts=False)
            assert unfused["speedup"] < fused["speedup"] <= fused["amdahl_limit"] + 1e-9
            assert 0 < fused["linear_share_bf16"] < 1
