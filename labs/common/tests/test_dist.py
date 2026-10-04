"""Tests for frontierlab.dist (Module 9). Multi-process tests use gloo on CPU with 2-4 ranks."""

import math

import pytest
import torch

from frontierlab.calc import from_hf, param_counts
from frontierlab.dist import goodput as G
from frontierlab.dist import layout as LY
from frontierlab.dist import recovery as R
from frontierlab.dist import ring_attention as RA
from frontierlab.dist import schedule as S
from frontierlab.perf.dist import spawn


# ---- layout planner ------------------------------------------------------------------------------------

def test_stage_params_sum_to_model():
    for spec in (LY.llama3_405b(), from_hf("deepseek-ai/DeepSeek-V3")):
        pc = param_counts(spec)
        for pp in (1, 4, 16):
            st = LY.stage_params(spec, pp)
            assert sum(len(r) for r in LY.stage_layers(spec.num_layers, pp)) == spec.num_layers
            assert sum(r["dense"] + r["expert"] for r in st) == pc["total"]
    assert param_counts(LY.llama3_405b())["total"] == pytest.approx(405.85e9, rel=1e-3)


def test_placement_llama3_order():
    lay = LY.Layout(tp=8, cp=1, pp=16, dp=128)            # Llama 3 Table 4, 16,384 GPUs
    assert lay.world == 16384
    assert LY.group_span(lay, ("tp",), 8) == 1             # TP stays inside a node
    assert LY.group_span(lay, ("pp",), 8) == 16
    assert LY.group_span(lay, LY.GROUPS["dp"], 8) == 128
    bad = LY.Layout(tp=8, pp=16, dp=128, order=("dp", "pp", "tp", "cp", "ep"))
    assert LY.group_span(bad, ("tp",), 8) == 8             # TP outermost: every TP group leaves the node
    assert sorted(LY.group_ranks(lay, ("tp",), rank=9)) == list(range(8, 16))


def test_activation_accounting_matches_korthikanti_for_gpt_layer():
    # MHA, I = 2h with SwiGLU's 4 saved (s, I) tensors = the 8h of a 4h GeLU MLP: 16h elements = 32·s·b·h bytes
    from frontierlab.calc import ArchSpec
    h = 1024
    spec = ArchSpec(name="gpt", model_type="x", vocab_size=1000, hidden_size=h, num_layers=2, layer_kinds=["full"] * 2,
                    num_heads=16, num_kv_heads=16, head_dim=64, v_head_dim=64, dense_intermediate=2 * h)
    tr = LY.Train(seq=2048, micro_batch=1, act_bytes=2)
    for tp in (1, 2, 8):
        got = LY.activation_bytes_per_layer(spec, LY.Layout(tp=tp), tr, moe_layer=False)
        assert got == pytest.approx(32 * 2048 * 1 * h / tp)
    full = LY.activation_bytes_per_layer(spec, LY.Layout(tp=2), LY.Train(seq=2048, recompute="full"), False)
    assert full == pytest.approx(2 * 2048 * h / 2)


def test_comm_formulas_by_hand():
    from frontierlab.calc import ArchSpec
    h = 64
    spec = ArchSpec(name="t", model_type="x", vocab_size=100, hidden_size=h, num_layers=4, layer_kinds=["full"] * 4,
                    num_heads=4, num_kv_heads=4, head_dim=16, v_head_dim=16, dense_intermediate=128)
    tr = LY.Train(seq=8, micro_batch=1, n_micro=2, act_bytes=2, comm_g_bytes=4)
    c = LY.comm_per_step(spec, LY.Layout(tp=2), tr)
    assert c["tp"] == pytest.approx(8 * 0.5 * 8 * 1 * h * 2 * 4 * 2)          # 8(t-1)/t · s b h e · layers · m
    n = param_counts(spec)["total"]
    assert LY.comm_per_step(spec, LY.Layout(dp=4, zero=0), tr)["dp"] == pytest.approx(2 * 3 / 4 * n * 4)
    assert LY.comm_per_step(spec, LY.Layout(dp=4, zero=1), tr)["dp"] == pytest.approx(3 / 4 * n * (4 + 2))
    assert LY.comm_per_step(spec, LY.Layout(dp=4, zero=3), tr)["dp"] == pytest.approx(3 / 4 * n * (4 + 2 * 2 * 2))
    c = LY.comm_per_step(spec, LY.Layout(cp=4), tr)
    assert c["cp"] == pytest.approx(3 * 3 * (8 / 4) * 4 * 32 * 2 * 4 * 2)        # 3(cp-1)·KV_local·layers·m


def test_zero_and_ep_memory():
    ds = from_hf("deepseek-ai/DeepSeek-V3")
    tr = LY.Train(seq=4096, n_micro=32, schedule="dualpipe", optim_bytes=4)
    lay = LY.Layout(pp=16, dp=128, ep=64, zero=1)
    m = {z: LY.memory_per_device(ds, LY.with_(lay, zero=z), tr) for z in (0, 1, 2, 3)}
    assert m[0]["total"] > m[1]["total"] > m[2]["total"] > m[3]["total"]
    ll = LY.llama3_405b()                                       # dense: the most loaded stage does not move
    d = {z: LY.memory_per_device(ll, LY.Layout(tp=8, pp=16, dp=64, zero=z), LY.Train()) for z in (0, 1, 3)}
    assert d[1]["optimizer"] == pytest.approx(d[0]["optimizer"] / 64)
    assert d[1]["weights"] == d[0]["weights"] and d[3]["weights"] == pytest.approx(d[0]["weights"] / 64)
    assert LY.check(ds, LY.Layout(pp=16, dp=128, ep=48), tr)   # 48 does not divide 128 or 256 experts
    r = LY.plan(ds, lay, tr, LY.H800_NODE)
    assert r["comm_bytes"]["ep"] > 0 and r["comm_bytes"]["tp"] == 0


# ---- ring attention ------------------------------------------------------------------------------------

def test_merge_and_simulated_ring_equal_full_attention():
    g = torch.Generator().manual_seed(0)
    T, n = 16, 4
    q = torch.randn(1, 4, T, 8, generator=g, dtype=torch.float64)
    k = torch.randn(1, 2, T, 8, generator=g, dtype=torch.float64)
    v = torch.randn(1, 2, T, 8, generator=g, dtype=torch.float64)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    for balanced in (False, True):
        pos = [RA.shard_positions(T, n, r, balanced) for r in range(n)]
        outs = RA.simulate_ring(q, k, v, pos)
        for r in range(n):
            assert (outs[r] - ref[:, :, pos[r]]).abs().max() < 1e-13


def test_load_balanced_sharding_equalises_work():
    assert len(set(RA.work_per_rank(64, 4, balanced=True))) == 1
    w = RA.work_per_rank(64, 4, balanced=False)
    assert w == sorted(w) and w[-1] / w[0] > 6                 # the last contiguous rank does ~7x the first


@pytest.mark.parametrize("world,balanced,T", [(2, True, 24), (4, False, 16)])
def test_ring_attention_forward_backward_float64(world, balanced, T):
    res = spawn(RA.equivalence_worker, world, T=T, balanced=balanced, H=4, KV=2, d=8)
    for r in res:
        assert max(r["out"], r["dq"], r["dk"], r["dv"]) < 1e-12, r


# ---- pipeline schedules --------------------------------------------------------------------------------

@pytest.mark.parametrize("t", [S.Times(1, 1, 1), S.Times(1, 2, 1), S.Times(2, 1, 0.5)])
def test_simulator_matches_published_bubbles(t):
    p, m = 8, 16
    rows = {r["key"]: r for r in S.compare(p, m, t, v=2)}
    for key in ("gpipe", "1f1b", "interleaved", "zb1p"):
        assert rows[key]["idle"] == pytest.approx([S.bubble_formula(key, p, m, t, 2 if key == "interleaved" else 1)] * p)
    assert rows["1f1b"]["bubble_ratio"] == pytest.approx(S.bubble_ratio_formula(p, m))
    assert rows["1f1b"]["peak_act"] == [p - s for s in range(p)]
    assert rows["gpipe"]["peak_act"] == [m] * p
    assert max(rows["zb1p"]["peak_act"]) == p                   # no device above 1F1B's worst
    assert rows["dualpipe"]["peak_act"] == [p + 1] * p          # DualPipe README: PP + 1
    assert rows["dualpipev"]["peak_act"] == [p + 1] * (p // 2)


def test_dualpipe_matches_readme_formula_when_f_b_w_equal():
    for p, m in ((4, 8), (8, 16), (8, 20)):
        t = S.Times(1, 1, 1)
        for sch, key in ((S.dualpipe(p, m), "dualpipe"), (S.dualpipev(p // 2, m), "dualpipev")):
            r = S.simulate(sch, t)
            assert r["idle"] == pytest.approx([S.bubble_formula(key, p, m, t)] * sch.devices)


def test_dualpipe_runs_every_microbatch_once():
    sch = S.dualpipe(8, 20)
    for d, ops in enumerate(sch.ops):
        f = [op.f for op in ops if op.f is not None]
        b = [op.b for op in ops if op.b is not None and op.kind in ("B", "Bz", "FB")]
        assert len(f) == len(set(f)) == 20 and len(b) == len(set(b)) == 20    # 10 per direction, 2 chunks


def test_one_f_one_b_order():
    assert S.one_f_one_b_order(4, 0, 6) == [("F", 0), ("F", 1), ("F", 2), ("F", 3), ("B", 0), ("F", 4), ("B", 1),
                                            ("F", 5), ("B", 2), ("B", 3), ("B", 4), ("B", 5)]


def test_real_pipeline_matches_single_process():
    from frontierlab.dist.pipeline import pipeline_worker
    res = spawn(pipeline_worker, 3, schedule="1f1b", m=4, steps=1, dtype="float64", check=True, hidden=32, vocab=61,
                seq=16, layers=3)
    for r in res:
        assert r["grad_max_abs_err"] < 1e-12
    assert res[-1]["loss_abs_err"] < 1e-12
    assert [r["peak_live_microbatches"] for r in res] == [3, 2, 1]


# ---- capability probes ---------------------------------------------------------------------------------

def test_tp_probe_finds_partial_qk_norm_gradients():
    from frontierlab.dist.capability import capability_worker
    names = ["TP (DTensor), course attention unchanged", "TP (DTensor) + per-rank head counts",
             "TP + per-rank heads + QK-norm grad all-reduce"]
    rows = {r["feature"]: r["status"] for r in spawn(capability_worker, 2, only=names)[0]["rows"]}
    assert rows == dict(zip(names, ["failed", "failed", "composed"]))


# ---- recovery ------------------------------------------------------------------------------------------

BASE = dict(steps=8, ckpt_every=3, global_batch=8, seq=16, hidden=32)


@pytest.fixture(scope="module")
def straight(tmp_path_factory):
    run = tmp_path_factory.mktemp("rec") / "straight"
    assert R.launch(2, run_dir=str(run), **BASE)["ok"]
    return run


def test_crash_and_resume_is_bit_identical(straight, tmp_path):
    run = tmp_path / "crash"
    first = R.launch(2, run_dir=str(run), die_at=5, **BASE)
    assert not first["ok"]                                      # rank 1 died with exit code 17
    assert [p.name for p in R.committed(run / "ckpt")] == ["step_000003"]
    assert R.launch(2, run_dir=str(run), **BASE)["ok"]
    starts = [r for r in R.read_metrics(run) if r["event"] == "start"]
    assert [s["from_step"] for s in starts] == [0, 3]
    cmp = R.compare_states(R.load_final(straight), R.load_final(run))
    assert all(v["bitwise_equal"] for v in cmp.values()), cmp
    assert R.losses_by_step(straight) == R.losses_by_step(run)


def test_async_checkpoints_do_not_change_the_run(straight, tmp_path):
    run = tmp_path / "async"
    assert R.launch(2, run_dir=str(run), async_save=True, **BASE)["ok"]
    cmp = R.compare_states(R.load_final(straight), R.load_final(run))
    assert all(v["bitwise_equal"] for v in cmp.values()), cmp
    assert [p.name for p in R.committed(run / "ckpt")] == ["step_000003", "step_000006", "step_000008"]


def test_crash_during_save_skips_uncommitted(tmp_path):
    run = tmp_path / "save-crash"
    assert not R.launch(2, run_dir=str(run), die_at=6, die_in_save=True, **BASE)["ok"]
    assert (run / "ckpt" / "step_000006").exists() and not (run / "ckpt" / "step_000006" / "COMMITTED").exists()
    assert R.latest_committed(run / "ckpt").name == "step_000003"


def test_reshard_fsdp2_to_other_layouts(straight, tmp_path):
    src = straight / "ckpt" / "step_000006"
    states = {}
    for world, mode in ((2, "fsdp2"), (4, "fsdp2"), (1, "ddp")):
        run = tmp_path / f"load-{mode}{world}"
        assert R.launch(world, run_dir=str(run), resume_from=str(src), mode=mode, **{**BASE, "steps": 6})["ok"]
        states[(world, mode)] = R.load_final(run)
        notes = [r["notes"] for r in R.read_metrics(run) if r["event"] == "start"][0]
        assert bool(notes) == (world != 2)                      # stateful per-rank RNG cannot follow a new world size
    ref = states[(2, "fsdp2")]
    for key in ((4, "fsdp2"), (1, "ddp")):
        cmp = R.compare_states(ref, states[key])
        for comp in ("model", "optim", "scheduler", "data", "step"):
            assert cmp[comp]["bitwise_equal"], (key, comp, cmp[comp])


def test_counter_rng_reshard_continues_close(tmp_path):
    a, b = tmp_path / "w2", tmp_path / "w4"
    kw = {**BASE, "rng_mode": "counter"}
    assert R.launch(2, run_dir=str(a), **kw)["ok"]
    assert R.launch(4, run_dir=str(b), resume_from=str(a / "ckpt" / "step_000003"), **kw)["ok"]
    la, lb = R.losses_by_step(a), R.losses_by_step(b)
    diffs = [abs(la[s] - lb[s]) for s in range(4, 9)]
    assert max(diffs) < 1e-9                                     # same data and noise; only the reduction order differs


# ---- goodput -------------------------------------------------------------------------------------------

def test_goodput_formulas():
    M = G.mtbf(54 * 24, 419)
    assert M == pytest.approx(3.093, abs=1e-3)
    assert G.young(1 / 60, M) == pytest.approx(math.sqrt(2 * M / 60))
    tau = G.optimal_tau(1 / 60, M, 10 / 60)
    assert G.daly(1 / 60, M) == pytest.approx(tau, rel=0.05)
    assert G.goodput(tau, 1 / 60, M, 10 / 60) >= G.goodput(G.young(1 / 60, M), 1 / 60, M, 10 / 60) - 1e-12
    assert G.job_mtbf(G.device_mtbf(M, 16384), 2 * 16384) == pytest.approx(M / 2)


def test_expected_wall_matches_monte_carlo():
    ana = G.expected_wall(10, 0.5, 0.05, 3.0, 0.2)
    mc = G.simulate_wall(10, 0.5, 0.05, 3.0, 0.2, runs=3000, seed=1)
    assert mc == pytest.approx(ana, rel=0.02)
