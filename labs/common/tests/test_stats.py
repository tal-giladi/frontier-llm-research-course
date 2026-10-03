import numpy as np

from frontierlab.stats import bootstrap_ci, holm, min_detectable_effect, paired_bootstrap, summary


def test_summary():
    s = summary([1.0, 2.0, 3.0])
    assert s["mean"] == 2.0 and abs(s["std"] - 1.0) < 1e-12


def test_paired_beats_unpaired_on_shared_difficulty():
    rng = np.random.default_rng(0)
    difficulty = rng.normal(0, 1.0, 500)
    a = difficulty + 0.05 + rng.normal(0, 0.05, 500)
    b = difficulty + rng.normal(0, 0.05, 500)
    p = paired_bootstrap(a, b)
    assert p["ci"][0] > 0                               # paired: clearly positive
    _, lo, hi = bootstrap_ci(a - 0)                     # unpaired spread is dominated by difficulty
    assert hi - lo > 10 * (p["ci"][1] - p["ci"][0])


def test_mde_and_holm():
    assert abs(min_detectable_effect(0.01, 3) - 2.8 * 0.01 * (2 / 3) ** 0.5) < 1e-12
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
