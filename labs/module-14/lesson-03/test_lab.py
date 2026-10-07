import math

import numpy as np
import torch

from frontierlab.labkit import load_target
from frontierlab.rlscale import asyncsim, mismatch

lab = load_target(__file__)


def test_schedule_by_hand():
    r0 = lab.simulate([2, 2, 2], 1.0, 0)
    assert r0["total_s"] == 9.0 and r0["lags"] == [0, 0, 0]
    r1 = lab.simulate([2, 2, 2], 1.0, 1)
    assert r1["total_s"] == 7.0 and r1["lags"] == [0, 1, 1]


def test_schedule_matches_reference_and_respects_bound():
    rng = np.random.default_rng(1)
    for _ in range(5):
        g = rng.lognormal(0, 1, size=40)
        T = float(rng.uniform(0.2, 3))
        for k in (0, 1, 2, 4, 8):
            mine, ref = lab.simulate(g, T, k), asyncsim.simulate(g, T, k)
            assert math.isclose(mine["total_s"], ref["total_s"], rel_tol=1e-12) and mine["lags"] == ref["lags"]
            assert mine["max_lag"] <= k


def test_mismatch_stats():
    torch.manual_seed(0)
    t = torch.randn(4, 6) * 0.1 - 1
    s = t + torch.randn(4, 6) * 0.01
    m = (torch.arange(6)[None] < torch.tensor([6, 3, 1, 4])[:, None]).float()
    a, b = lab.mismatch_stats(t, s, m), mismatch.mismatch_stats(t, s, m)
    for k in b:
        assert math.isclose(a[k], b[k], rel_tol=1e-9, abs_tol=1e-12), k


def test_fixed_tile_matmul_is_batch_invariant():
    torch.manual_seed(0)
    W, x = torch.randn(512, 256), torch.randn(130, 512)
    full = lab.fixed_tile_matmul(x, W)
    assert full.shape == (130, 256) and torch.allclose(full, x @ W, atol=1e-3)
    for b in (1, 3, 64, 65):
        assert torch.equal(lab.fixed_tile_matmul(x[:b], W)[0], full[0])
