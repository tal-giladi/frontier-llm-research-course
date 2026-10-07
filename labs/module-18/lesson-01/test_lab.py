import math

import numpy as np

from frontierlab.alignment import monitors as M
from frontierlab.labkit import load_target
from frontierlab.stats import paired_bootstrap

lab = load_target(__file__)


def test_wilson_matches_reference():
    for k, n in [(0, 100), (28, 100), (59, 100), (100, 100), (3, 7)]:
        got, want = lab.wilson(k, n), M.wilson(k, n)
        assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(got, want))
    assert all(math.isnan(x) for x in lab.wilson(0, 0))
    p, lo, hi = lab.wilson(20, 100)
    assert math.isclose(lo, 0.1334, abs_tol=1e-3) and math.isclose(hi, 0.2888, abs_tol=1e-3)


def test_persona_position():
    assert lab.persona_position(2.0, 1.0, 3.0) == 0.5
    assert lab.persona_position(1.0, 1.0, 3.0) == 0.0
    assert lab.persona_position(4.0, 1.0, 3.0) == 1.5
    assert lab.persona_position(-1.0, 1.0, -3.0) == 0.5        # the axis may point either way


def test_paired_shift_matches_course_bootstrap():
    rng = np.random.default_rng(3)
    a = rng.random(80)
    b = a - 0.1 + 0.02 * rng.standard_normal(80)
    m, lo, hi = lab.paired_shift(a.tolist(), b.tolist(), n_boot=4000)
    ref = paired_bootstrap(a, b, n_boot=4000)
    assert math.isclose(m, ref["mean_diff"], rel_tol=1e-12)
    assert lo < m < hi and abs(lo - ref["ci"][0]) < 0.01 and abs(hi - ref["ci"][1]) < 0.01
    assert lo > 0.09


def test_em_verdict():
    s = {"sycophancy": (0.10, 0.06, 0.14), "copying": (0.08, 0.02, 0.13), "unsafe_pattern": (0.9, 0.85, 0.95)}
    assert lab.em_verdict(s, ("sycophancy", "copying")) == "broad"
    s2 = {**s, "copying": (0.01, -0.02, 0.04)}
    assert lab.em_verdict(s2, ("sycophancy", "copying")) == "mixed"
    s3 = {**s2, "sycophancy": (0.05, -0.01, 0.10)}
    assert lab.em_verdict(s3, ("sycophancy", "copying")) == "narrow"
    assert lab.em_verdict({"x": (0.02, 0.01, 0.03)}, ("x",)) == "narrow"   # real but below the minimum effect
    assert lab.MIN_EFFECT == 0.03
