import numpy as np
import pytest
from scipy import stats as st

from frontierlab.labkit import load_target
from frontierlab.stats import min_detectable_effect, paired_bootstrap

lab = load_target(__file__)


def test_mde_matches_hand_value_and_shared_code():
    # hand example in the lesson: seed std 0.01, 3 seeds per arm -> about 0.0229
    assert lab.mde(0.01, 3) == pytest.approx(0.022875, abs=2e-5)
    assert lab.mde(0.01, 3) == pytest.approx(min_detectable_effect(0.01, 3), rel=2e-3)
    assert lab.mde(0.02, 8) == pytest.approx(lab.mde(0.01, 2))


def test_seeds_needed():
    assert lab.seeds_needed(0.01, 0.0229) == 3
    assert lab.seeds_needed(0.01, 0.02) == 4
    assert lab.seeds_needed(0.01, 0.01) == 16


def test_unpaired_is_wider_than_paired_when_items_share_difficulty():
    rng = np.random.default_rng(1)
    difficulty = rng.normal(5.0, 0.5, 256)
    a = difficulty + 0.02 + rng.normal(0, 0.01, 256)
    b = difficulty + rng.normal(0, 0.01, 256)
    u = lab.unpaired_bootstrap(a, b)
    p = paired_bootstrap(a, b)
    assert u["mean_diff"] == pytest.approx(p["mean_diff"])
    assert u["ci"][0] < 0 < u["ci"][1]            # unpaired cannot see a 0.02 gain under 0.5 spread
    assert p["ci"][0] > 0                          # paired sees it clearly
    assert (u["ci"][1] - u["ci"][0]) > 20 * (p["ci"][1] - p["ci"][0])


def test_seed_level_paired_matches_scipy():
    base = np.array([3.10, 3.14, 3.08])
    new = np.array([3.09, 3.12, 3.075])
    r = lab.seed_level_ci(base, new, paired=True)
    res = st.ttest_rel(new, base).confidence_interval()
    assert r["ci"] == pytest.approx((res.low, res.high))


def test_seed_level_welch_matches_scipy():
    base = np.array([3.10, 3.14, 3.08, 3.12, 3.11])
    new = np.array([3.09, 3.12, 3.075])
    r = lab.seed_level_ci(base, new, paired=False)
    res = st.ttest_ind(new, base, equal_var=False).confidence_interval()
    assert r["ci"] == pytest.approx((res.low, res.high))
