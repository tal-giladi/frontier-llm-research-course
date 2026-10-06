"""Correctness checks for frontierlab.scaling (Module 11). No data downloads; one tiny training run."""

import json
import math

import numpy as np
import pytest
import torch

from frontierlab.model import LM, PRESETS
from frontierlab.scaling import derisk, downstream, ladder, laws
from frontierlab.scaling import fit as sfit


def test_chinchilla_closed_form_matches_grid():
    C = 5.76e23                                         # Gopher / Chinchilla budget
    opt = laws.compute_optimal(C)
    N = np.geomspace(1e9, 1e12, 20001)
    L = laws.loss(N, C / (6 * N))
    assert abs(N[np.argmin(L)] / opt["N"] - 1) < 1e-3
    assert abs(opt["exp_N"] - 0.28 / 0.62) < 1e-12 and abs(opt["exp_N"] + opt["exp_D"] - 1) < 1e-12


def test_compute_for_loss_inverts_optimal_loss():
    for C in (1e20, 1e23):
        L = float(laws.optimal_loss(C))
        assert abs(laws.compute_for_loss(L) / C - 1) < 1e-6


def test_overtraining_is_zero_at_the_optimum_and_positive_off_it():
    opt = laws.compute_optimal(1e22)
    assert abs(laws.overtraining(float(opt["N"]), float(opt["D"]))["overhead"]) < 1e-6
    small = laws.overtraining(float(opt["N"]) / 8, float(opt["D"]) * 8)
    assert small["overhead"] > 0 and small["loss"] > small["loss_opt_same_C"]


def test_tokens_for_loss_round_trip_and_unreachable():
    D = laws.tokens_for_loss(1e9, float(laws.loss(1e9, 2e11)))
    assert abs(float(D) / 2e11 - 1) < 1e-9
    assert np.isinf(laws.tokens_for_loss(1e6, 1.70))   # a 1M model cannot reach E + 0.01


def test_inference_aware_prefers_smaller_models_with_more_inference():
    L = float(laws.optimal_loss(1e22))
    a = laws.inference_aware(L, 0.0)
    b = laws.inference_aware(L, 1e13)
    assert b["N"] < a["N"] and b["tokens_per_param"] > a["tokens_per_param"]
    assert b["total_flops"] <= b["total_flops_if_train_only_optimal"] + 1e-6 * b["total_flops"]


def test_effective_data():
    assert float(laws.effective_data(100.0, 50.0)) == 50.0
    # 4 epochs: R = 3 repeats -> U + U R* (1 - e^{-3/R*}) ~ 3.73 U, close to 4 U
    d4 = float(laws.effective_data(100.0, 400.0))
    assert 360 < d4 < 400
    assert float(laws.effective_data(100.0, 1e6)) < 100 * (1 + laws.MUENNIGHOFF["R_D_star"]) + 1e-9


def test_parametric_fit_recovers_a_planted_law():
    rng = np.random.default_rng(0)
    true = {"E": 2.0, "A": 30.0, "B": 400.0, "alpha": 0.35, "beta": 0.45}
    N = np.repeat(np.geomspace(1e4, 1e6, 5), 4)
    D = np.tile(np.geomspace(1e5, 1e7, 4), 5)
    L = sfit.predict(true, N, D) * np.exp(rng.normal(0, 1e-4, N.size))
    f = sfit.fit_parametric(N, D, L)
    assert abs(f["alpha"] - 0.35) < 0.02 and abs(f["beta"] - 0.45) < 0.02 and abs(f["E"] - 2.0) < 0.05
    assert f["rms_log"] < 1e-3
    held = sfit.holdout(N, D, L, N == N.max())
    assert held["max_abs"] < 0.01


def test_fixed_exponent_fit_is_linear_least_squares():
    N, D = np.geomspace(1e4, 1e6, 6), np.geomspace(1e6, 1e5, 6)
    L = 1.5 + 20 * N ** -0.3 + 50 * D ** -0.4
    f = sfit.fit_parametric(N, D, L, fix={"alpha": 0.3, "beta": 0.4})
    assert abs(f["E"] - 1.5) < 1e-8 and abs(f["A"] - 20) < 1e-6 and abs(f["B"] - 50) < 1e-6


def test_isoflop_minima_find_the_planted_optimum():
    c = {"E": 2.0, "A": 50.0, "B": 500.0, "alpha": 0.4, "beta": 0.4}
    runs = []
    for C in (1e12, 1e13, 1e14):
        for N in np.geomspace(laws.compute_optimal(C, c)["N"] / 8, laws.compute_optimal(C, c)["N"] * 8, 7):
            D = C / (6 * N)
            runs.append({"budget": C, "C": C, "N": N, "D": D, "loss": float(sfit.predict(c, N, D))})
    mins = sfit.isoflop_minima(runs)
    for m in mins:
        assert abs(math.log10(m["N_opt"] / laws.compute_optimal(m["budget"], c)["N"])) < 0.05 and not m["edge"]
    a = sfit.fit_allocation([m["budget"] for m in mins], [m["N_opt"] for m in mins])["a"]
    assert abs(a - 0.5) < 0.02


def test_power_offset_fit():
    C = np.geomspace(1e10, 1e14, 8)
    L = 1.8 + 40 * C ** -0.12
    f = sfit.fit_power_offset(C, L, E_grid=np.linspace(0, 1.79, 1791))
    assert abs(f["E"] - 1.8) < 0.02 and abs(f["gamma"] - 0.12) < 0.01


def test_bootstrap_interval_covers_the_truth():
    rng = np.random.default_rng(1)
    true = {"E": 2.0, "A": 30.0, "B": 400.0, "alpha": 0.35, "beta": 0.45}
    N = np.repeat(np.geomspace(1e4, 1e6, 4), 3)
    D = np.tile(np.geomspace(1e5, 1e7, 3), 4)
    L = sfit.predict(true, N, D) + rng.normal(0, 0.01, N.size)
    fits = sfit.bootstrap(N, D, L, n_boot=60, seed=0, alphas=np.linspace(0.2, 0.6, 9), betas=np.linspace(0.2, 0.7, 11))
    mid, lo, hi = sfit.prediction_interval(fits, 3e6, 3e7, level=0.95)
    assert lo <= float(sfit.predict(true, 3e6, 3e7)) + 0.05 and hi >= float(sfit.predict(true, 3e6, 3e7)) - 0.05 and lo < hi


def test_ladder_presets_registered_and_plans():
    assert "m11-r3" in PRESETS and "m11-350m" in PRESETS
    plan = ladder.plan_isoflop([1e12], ["m11-r1", "m11-r3"], seq=128, batch=8)
    for p in plan:
        assert abs(p["C"] / 1e12 - 1) < 0.02
        assert p["fpt"] == ladder.fpt(ladder.config(p["preset"]), 128)
    s = ladder.sizes(ladder.config("m11-r3"))
    assert s["N_total"] == sum(q.numel() for q in LM(ladder.config("m11-r3")).parameters())
    assert 3.0e8 < ladder.sizes(ladder.config("m11-350m"))["N_total"] < 4.0e8
    fr = ladder.plan_fixed_ratio(["m11-r2"], [5, 20], seq=128, batch=16)
    n = ladder.sizes(ladder.config("m11-r2"))["N_total"]
    assert [abs(p["tokens"] / (r * n) - 1) < 2048 / (r * n) for p, r in zip(fr, (5, 20))] == [True, True]


def test_sigmoid_fit_recovers_parameters():
    x = np.linspace(4, 8, 15)
    y = downstream.sigmoid_curve(x, 6.0, 2.0, 0.25, 1.0)
    f = downstream.fit_sigmoid(x, y)
    assert abs(f["x0"] - 6.0) < 1e-3 and abs(f["s"] - 2.0) < 1e-3


def test_metrics_from_logprobs_by_hand():
    lp = torch.tensor([[0.0, -1.0, -2.0, -3.0], [-2.0, -1.0, 0.0, -3.0]], dtype=torch.float64)
    m = downstream.metrics_from_logprobs(lp, torch.tensor([0, 0]), cont=2)
    p = torch.softmax(lp, 1)
    assert m["acc"] == 0.5 and abs(m["p_correct"] - float(p[:, 0].mean())) < 1e-12
    assert abs(m["nll_correct"] - (0.0 + 2.0) / 2 / 2) < 1e-12


class _Data:
    def __init__(self, n=4000, docs=40, seed=0):
        g = np.random.default_rng(seed)
        self.tokens = g.integers(1, 50, n).astype(np.uint16)
        self.doc_starts = np.arange(0, n, n // docs)


def test_cloze_items_shapes_and_answers():
    items = downstream.build_cloze(_Data(), n_items=50, ctx=12, cont=4, seed=3)
    assert items["context"].shape == (50, 12) and items["choices"].shape == (50, 4, 4)
    tok = _Data().tokens
    # the answer option continues the context in the source text
    starts = [int(np.flatnonzero(np.all(np.lib.stride_tricks.sliding_window_view(tok, 12) == c.numpy(), axis=1))[0])
              for c in items["context"][:5]]
    for i, s in enumerate(starts):
        assert np.array_equal(items["choices"][i, items["answer"][i]].numpy(), tok[s + 12:s + 16])
    counts = np.bincount(items["answer"].numpy(), minlength=4)
    assert counts.min() > 3


def test_option_logprobs_match_a_direct_computation():
    torch.manual_seed(0)
    cfg = PRESETS["m11-r1"](vocab_size=50)
    m = LM(cfg).double()
    items = downstream.build_cloze(_Data(), n_items=6, ctx=8, cont=3, seed=1)
    lp = downstream.option_logprobs(m, items)
    x = torch.cat([items["context"][2], items["choices"][2, 1]])[None]
    with torch.no_grad():
        logp = torch.log_softmax(m(x).logits.double(), -1)[0]
    direct = sum(float(logp[8 - 1 + j, x[0, 8 + j]]) for j in range(3))
    assert abs(float(lp[2, 1]) - direct) < 1e-10
    em = downstream.exact_match(m, items, k=2)
    assert 0 <= em["exact_match"] <= em["token_acc"] <= 1 or em["token_acc"] == 0


def test_pca_recovers_a_rank_one_structure():
    rng = np.random.default_rng(0)
    s = rng.normal(size=40)
    X = np.outer(s, rng.uniform(0.5, 2, 6)) + rng.normal(0, 0.01, (40, 6))
    out = downstream.pca_capabilities(X, 2)
    assert out["explained"][0] > 0.99
    assert abs(np.corrcoef(out["S"][:, 0], s)[0, 1]) > 0.999


def test_logistic_fit():
    S = np.linspace(-3, 3, 30)[:, None]
    y = 0.25 + 0.75 / (1 + np.exp(-(2 * S[:, 0] - 1)))
    f = downstream.fit_logistic(S, y, lo=0.25, hi=1.0, l2=0.0)
    assert abs(f["w"][0] - 2) < 1e-3 and abs(f["b"] + 1) < 1e-3


def test_preregistration_is_write_once_and_tamper_evident(tmp_path):
    p = tmp_path / "pred.json"
    derisk.preregister(p, {"prediction": {"loss": 4.0, "lo": 3.9, "hi": 4.1, "level": 0.9}})
    with pytest.raises(FileExistsError):
        derisk.preregister(p, {"prediction": {"loss": 5.0, "lo": 4.9, "hi": 5.1}})
    d = json.loads(p.read_text())
    d["body"]["prediction"]["loss"] = 3.95
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        derisk.load_prereg(p)


def test_curve_band_and_monitor():
    true = {"E": 2.0, "A": 30.0, "B": 400.0, "alpha": 0.35, "beta": 0.45}
    runs = []
    for i, (N, D) in enumerate([(1e4, 1e6), (3e4, 1e6), (1e5, 3e5), (3e4, 3e6), (1e5, 1e6), (3e5, 3e5)]):
        curve = [(int(f * 100), float(sfit.predict(true, N, D)) + 1.5 * (1 - f)) for f in np.linspace(0.1, 1, 10)]
        runs.append({"run": f"r{i}", "N_total": N, "D": D, "steps": 100, "val_curve": curve})
    band = derisk.curve_band(runs, 1e6, 3e6, [0.1, 0.2, 0.5, 1.0], 0.35, 0.45, n_boot=50)
    exp_final = float(sfit.predict(true, 1e6, 3e6))
    assert abs(band[-1]["mid"] - exp_final) < 1e-6
    good = [(int(f * 200), float(sfit.predict(true, 1e6, 3e6)) + 1.5 * (1 - f)) for f in np.linspace(0.1, 1, 10)]
    assert derisk.monitor(good, 200, band)["decision"] == "continue"
    bad = [(s, v + 0.3) for s, v in good]
    m = derisk.monitor(bad, 200, band)
    assert m["decision"] == "stop" and m["off_band"]["step"] == 20 and abs(m["compute_saved"] - 0.9) < 1e-12
    spikes = [(s, 1.0) for s in range(0, 200, 10)] + [(150, 9.0)]
    assert derisk.monitor(good, 200, band, grad_norms=spikes)["grad_spike"]["step"] == 150


def test_wrapper_unique_tokens_cap_and_resume(tmp_path):
    """A capped run draws windows only from the first U tokens, records it, and resumes exactly."""
    from frontierlab.data.loader import TokenData
    from frontierlab.metrics.jsonl import read_jsonl
    from frontierlab.runcard import read_run_card
    from frontierlab.scaling import train as strain
    try:
        TokenData("train")
    except FileNotFoundError:
        pytest.skip("Data-v0 not prepared")
    common = ["--preset", "m11-r1", "--steps", "12", "--batch", "2", "--seq", "32", "--warmup", "2",
              "--log-every", "1", "--eval-every", "12", "--eval-windows", "4", "--ckpt-every", "100"]
    strain.main(["--unique-tokens", "4096", "--run", str(tmp_path / "a")] + common)
    strain.main(["--unique-tokens", "4096", "--run", str(tmp_path / "b"), "--stop-after", "5"] + common)
    strain.main(["--unique-tokens", "4096", "--run", str(tmp_path / "b")] + common)
    la = [r["loss"] for r in read_jsonl(tmp_path / "a" / "metrics.jsonl") if r["split"] == "train"]
    lb = {r["step"]: r["loss"] for r in read_jsonl(tmp_path / "b" / "metrics.jsonl") if r["split"] == "train"}
    assert la == [lb[s] for s in sorted(lb)]
    card = read_run_card(tmp_path / "a")
    assert card["scaling"]["unique_tokens"] == 4096 and abs(card["scaling"]["epochs"] - 12 * 64 / 4096) < 1e-12
    g = torch.Generator().manual_seed(0)
    strain.CappedTokenData.cap = 4096
    try:
        d = strain.CappedTokenData("train")
        x = d.batch(64, 32, g)
        ref = torch.from_numpy(np.asarray(d.tokens[:4096]).astype(np.int64))
        windows = ref.unfold(0, 32, 1)
        assert all(((windows == row).all(1)).any() for row in x)
    finally:
        strain.CappedTokenData.cap = None
