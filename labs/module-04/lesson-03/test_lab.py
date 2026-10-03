import numpy as np
import pytest
import torch

from frontierlab.flops import PEAK_BF16, flops_per_token
from frontierlab.labkit import load_target
from frontierlab.model import baseline0

lab = load_target(__file__)

RNG = np.random.default_rng(0)
LENGTHS = RNG.integers(3, 300, size=200)
STARTS = np.concatenate([[0], np.cumsum(LENGTHS)[:-1]]).astype(np.int64)


def test_same_doc_context_by_hand():
    starts = np.array([0, 5, 7])
    got = lab.same_doc_context(starts, np.array([3]), 6)          # tokens 3..8: docs 0,0,1,1,2,2
    assert got.tolist() == [[0, 1, 0, 1, 0, 1]]
    got = lab.same_doc_context(starts, np.array([0, 7]), 3)
    assert got.tolist() == [[0, 1, 2], [0, 1, 2]]


def test_same_doc_context_matches_reference():
    from frontierlab.data.loader import TokenData
    from frontierlab.longctx.data import same_doc_context as ref

    class Fake(TokenData):
        def __init__(self):
            self.tokens, self.doc_starts = np.zeros(int(LENGTHS.sum()), dtype=np.uint16), STARTS
    d = Fake()
    g = torch.Generator().manual_seed(4)
    ws = torch.randint(0, len(d.tokens) - 64, (32,), generator=g).numpy()
    assert np.array_equal(lab.same_doc_context(STARTS, ws, 64), ref(d, 64, n_windows=32, seed=4))


@pytest.mark.parametrize("T", [1, 50, 200, 299])
def test_within_doc_start_enumerates_every_window_once(T):
    ok = LENGTHS >= T
    total = int((LENGTHS[ok] - T + 1).sum())
    starts = [lab.within_doc_start(STARTS, LENGTHS, T, u) for u in range(total)]
    assert len(set(starts)) == total                              # a bijection onto valid starts
    doc = np.searchsorted(STARTS, np.array(starts), side="right") - 1
    assert (np.array(starts) + T <= STARTS[doc] + LENGTHS[doc]).all()   # never crosses the document end
    assert starts == sorted(starts)                               # file order


def test_stage_plan():
    cfg = baseline0()
    rows = lab.stage_plan(cfg, [(8192, 4e8), (32768, 2e8)], PEAK_BF16["H100-SXM"], 0.3)
    assert [r["seq"] for r in rows] == [8192, 32768, "total"]
    f1 = flops_per_token(cfg, 8192) * 4e8
    assert abs(rows[0]["flops"] - f1) / f1 < 1e-12
    assert abs(rows[1]["gpu_hours"] - flops_per_token(cfg, 32768) * 2e8 / (989e12 * 0.3) / 3600) < 1e-9
    assert abs(rows[2]["gpu_hours"] - rows[0]["gpu_hours"] - rows[1]["gpu_hours"]) < 1e-9
    assert rows[1]["flops"] / 2e8 > 2 * rows[0]["flops"] / 4e8          # attention: a 32K token costs > 2x an 8K token


def test_decide():
    assert lab.decide((0.00, 0.01), (0.03, 0.05), 0.02, 0.01) == "adopt"
    assert lab.decide((0.03, 0.05), (0.03, 0.05), 0.02, 0.01) == "reject"
    assert lab.decide((0.00, 0.01), (-0.01, 0.005), 0.02, 0.01) == "reject"
    assert lab.decide((0.01, 0.03), (0.03, 0.05), 0.02, 0.01) == "inconclusive"
    assert lab.decide((0.00, 0.01), (0.00, 0.05), 0.02, 0.01) == "inconclusive"
