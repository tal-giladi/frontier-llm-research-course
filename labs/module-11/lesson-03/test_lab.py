import math

import numpy as np

from frontierlab.labkit import load_target
from frontierlab.scaling import derisk

lab = load_target(__file__)


def test_interval():
    p = np.arange(1001, dtype=float)
    lo, hi = lab.interval(p, 0.9)
    assert abs(lo - 50.0) < 1e-9 and abs(hi - 950.0) < 1e-9


def test_loss_at_fraction_matches_reference():
    curve = [(10, 5.0), (20, 4.5), (30, 4.2), (40, 4.0), (50, 3.9), (100, 3.5)]
    for f in (0.1, 0.2, 0.31, 0.5, 1.0, 0.75):
        a, b = lab.loss_at_fraction(curve, 100, f), derisk.loss_at_fraction(curve, 100, f)
        assert (math.isnan(a) and math.isnan(b)) or a == b


def test_first_off_band_matches_monitor():
    band = [{"fraction": f, "lo": 0, "mid": 0, "hi": 4.0 - f} for f in (0.1, 0.5, 1.0)]
    good = [(s, 4.0 - s / 100 - 0.01) for s in range(10, 101, 10)]
    bad = [(s, v + (0.3 if s >= 40 else 0.0)) for s, v in good]
    assert lab.first_off_band(good, 100, band) is None
    assert lab.first_off_band(bad, 100, band) == 40 == derisk.monitor(bad, 100, band)["off_band"]["step"]
    early = [(5, 9.0)] + good                                   # before min_fraction: ignored
    assert lab.first_off_band(early, 100, band) is None


def test_lr_sensitivity():
    # losses 4.0 (best), 4.5, 9.0 (diverged above the initial 7.0): mean(min(.,7)) - 4.0 = (4 + 4.5 + 7)/3 - 4
    assert abs(lab.lr_sensitivity({1e-3: 4.5, 3e-3: 4.0, 1e-2: 9.0}, 7.0) - ((4 + 4.5 + 7) / 3 - 4)) < 1e-12


def test_go_no_go():
    assert lab.go_no_go({"a": (True, ""), "b": (True, "")}) == "go"
    assert lab.go_no_go({"z": (False, "x"), "a": (False, "y"), "m": (True, "")}) == "no-go: a, z"
