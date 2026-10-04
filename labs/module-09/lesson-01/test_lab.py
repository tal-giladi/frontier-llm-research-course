import math

import pytest
import torch

from frontierlab.dist import layout as LY
from frontierlab.dist import ring_attention as RA
from frontierlab.labkit import load_target

lab = load_target(__file__)

ORDER = ["tp", "cp", "pp", "dp"]


def test_group_nodes_matches_planner():
    for tp, cp, pp, dp in ((8, 1, 16, 128), (8, 16, 16, 8), (2, 2, 2, 2), (1, 1, 16, 128)):
        lay = LY.Layout(tp=tp, cp=cp, pp=pp, dp=dp, order=("tp", "cp", "pp", "ep", "dp"))
        sizes = {"tp": tp, "cp": cp, "pp": pp, "dp": dp}
        for dims in (["tp"], ["cp"], ["pp"], ["dp"], ["tp", "cp"]):
            want = LY.group_span(lay, tuple(dims), 8)
            assert lab.group_nodes(ORDER, sizes, dims, 8) == want, (sizes, dims)
    # the order matters: TP outermost on 16 GPUs scatters every TP group over both nodes
    assert lab.group_nodes(["dp", "tp"], {"tp": 8, "dp": 2}, ["tp"], 8) == 2
    assert lab.group_nodes(["tp", "dp"], {"tp": 8, "dp": 2}, ["tp"], 8) == 1


def test_state_bytes_zero_stages():
    n = 8e9
    assert lab.state_bytes_per_device(n, 1, 1, 8, 0) == pytest.approx(n * 18)
    assert lab.state_bytes_per_device(n, 1, 1, 8, 1) == pytest.approx(n * (2 + 4 + 12 / 8))
    assert lab.state_bytes_per_device(n, 1, 1, 8, 2) == pytest.approx(n * (2 + 16 / 8))
    assert lab.state_bytes_per_device(n, 1, 1, 8, 3) == pytest.approx(n * 18 / 8)
    assert lab.state_bytes_per_device(n, 8, 16, 128, 1, optim=4) == pytest.approx(n / 128 * (6 + 8 / 128))


def test_tp_cp_bytes_match_planner():
    from frontierlab.calc import ArchSpec
    h, L = 256, 2
    spec = ArchSpec(name="t", model_type="x", vocab_size=100, hidden_size=h, num_layers=L, layer_kinds=["full"] * L,
                    num_heads=8, num_kv_heads=4, head_dim=32, v_head_dim=32, dense_intermediate=512)
    tr = LY.Train(seq=64, micro_batch=2, n_micro=1, act_bytes=2)
    for tp, cp in ((2, 1), (4, 1), (1, 4), (2, 2)):
        c = LY.comm_per_step(spec, LY.Layout(tp=tp, cp=cp), tr)
        got = lab.tp_cp_bytes_per_layer(64, 2, h, (4 // tp) * 64, 2, tp, cp)
        assert got["tp"] * L == pytest.approx(c["tp"])
        assert got["cp"] * L == pytest.approx(c["cp"])


def test_merge_rule():
    g = torch.Generator().manual_seed(0)
    T, n = 16, 4
    q = torch.randn(1, 4, T, 8, generator=g, dtype=torch.float64)
    k = torch.randn(1, 2, T, 8, generator=g, dtype=torch.float64)
    v = torch.randn(1, 2, T, 8, generator=g, dtype=torch.float64)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    for balanced in (False, True):
        pos = [RA.shard_positions(T, n, r, balanced) for r in range(n)]
        outs = RA.simulate_ring(q, k, v, pos, merge_fn=lab.merge)
        for r in range(n):
            assert (outs[r] - ref[:, :, pos[r]]).abs().max() < 1e-13
    o = torch.zeros(1, 1, 2, 3, dtype=torch.float64)
    lse = torch.full((1, 1, 2), -math.inf, dtype=torch.float64)
    mo, ml = lab.merge(o, lse, o, lse)
    assert not torch.isnan(mo).any() and torch.isinf(ml).all()
