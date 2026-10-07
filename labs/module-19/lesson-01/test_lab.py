import math

import pytest

from frontierlab.labkit import load_target
from frontierlab.research import proposals as P

lab = load_target(__file__)


def test_score():
    assert math.isclose(lab.score(6, 0.5, 3.0), 1.0)
    assert math.isclose(lab.score(8, 0.18, 40.0), 0.036)


def test_p_decisive_is_power_times_transfer():
    for eff, sd, n, pt in [(0.03, 0.02, 3, 0.5), (0.05, 0.02, 2, 0.6), (0.0, 0.02, 3, 1.0), (0.5, 0.01, 5, 0.9)]:
        assert math.isclose(lab.p_decisive(eff, sd, n, pt), P.power(eff, sd, n) * pt, rel_tol=1e-12)


@pytest.mark.parametrize("sizes,effects,noise", [
    ([1, 2, 3], [0.01, -0.01, 0.005], 0.01), ([1, 2, 3], [0.05, 0.01, -0.06], 0.01),
    ([3, 1, 2], [0.10, 0.03, 0.06], 0.01), ([1, 2, 3], [0.10, 0.06, 0.03], 0.01),
    ([1, 2, 3], [0.05, 0.06, 0.055], 0.01), ([1e5, 4e5], [-0.3, -0.5], 0.03), ([2], [0.2], 0.05),
])
def test_proxy_trend_matches_shared(sizes, effects, noise):
    assert lab.proxy_trend(sizes, effects, noise) == P.proxy_trend(sizes, effects, noise)


def test_your_proposals_are_complete_and_ranked():
    props = [P.Proposal(d["claim_id"], d["question"], d["decision"], d["value"],
                        lab.p_decisive(d["effect"], d["seed_std"], d["n_seeds"], d["p_transfer"]),
                        d["cost_gpu_h"], d["proxy"], d["kill"]) for d in lab.PROPOSALS]
    assert P.check_proposals(props) == []
    scores = [P.score(p) for p in props]
    assert scores == sorted(scores, reverse=True), "list the proposals best first, by your score"
