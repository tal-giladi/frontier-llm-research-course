import math

import pytest

from frontierlab.dist import goodput as G
from frontierlab.labkit import load_target

lab = load_target(__file__)


def test_young_and_goodput():
    assert lab.young(1 / 60, 3.0) == pytest.approx(math.sqrt(0.1))
    for tau, d, M, R in ((0.3, 1 / 60, 3.09, 1 / 6), (1.0, 0.1, 2.0, 0.5), (0.05, 0.01, 50.0, 0.0)):
        assert lab.goodput(tau, d, M, R) == pytest.approx(G.goodput(tau, d, M, R))
    # no failures in sight: goodput -> tau / (tau + delta)
    assert lab.goodput(1.0, 0.1, 1e9, 0.0) == pytest.approx(1 / 1.1, rel=1e-6)


def test_latest_committed(tmp_path):
    root = tmp_path / "ckpt"
    assert lab.latest_committed(root) is None
    for s, done in ((3, True), (6, True), (9, False)):
        d = root / f"step_{s:06d}"
        d.mkdir(parents=True)
        if done:
            (d / "COMMITTED").write_text("{}")
    assert lab.latest_committed(root).name == "step_000006"


def test_lost_work():
    rec = [{"event": "start", "from_step": 0}] + [{"event": "step", "step": s} for s in range(1, 6)] + \
          [{"event": "start", "from_step": 3}] + [{"event": "step", "step": s} for s in range(4, 9)] + \
          [{"event": "start", "from_step": 6}, {"event": "start", "from_step": 6}]
    assert lab.lost_work(rec) == [2, 2, 0]
    assert lab.lost_work([{"event": "start", "from_step": 0}]) == []
