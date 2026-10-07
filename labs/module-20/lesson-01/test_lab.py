import math

import numpy as np
import pytest

from frontierlab.capstone import package as PK
from frontierlab.capstone import sample as SA
from frontierlab.capstone import uncertainty as U
from frontierlab.labkit import load_target

lab = load_target(__file__)


def test_hierarchical_interval_against_reference():
    rng = np.random.default_rng(3)
    for S, W, shift in ((3, 256, 0.01), (2, 64, -0.03), (5, 100, 0.0)):
        b = rng.normal(4.0, 0.3, size=(S, W))
        a = b + shift + rng.normal(0, 0.02, size=(S, 1)) + rng.normal(0, 0.05, size=(S, W))
        mine, ref = lab.hierarchical_interval(a, b), U.hierarchical_bootstrap(a, b)
        assert math.isclose(mine["mean_diff"], ref["mean_diff"], abs_tol=1e-12)
        width = ref["ci"][1] - ref["ci"][0]
        assert abs(mine["ci"][0] - ref["ci"][0]) < 0.15 * width and abs(mine["ci"][1] - ref["ci"][1]) < 0.15 * width
        assert mine["ci"][0] <= mine["mean_diff"] <= mine["ci"][1] and mine["method"]
    with pytest.raises(ValueError):
        lab.hierarchical_interval(np.zeros((3, 4)), np.zeros((3, 5)))


def test_hierarchical_interval_resamples_seeds():
    b = np.zeros((3, 300))
    a = b + np.array([0.0, 0.04, 0.08])[:, None]
    lo, hi = lab.hierarchical_interval(a, b)["ci"]
    assert hi - lo > 0.04, "the interval ignores seed-to-seed variation: resample seeds first"


@pytest.mark.parametrize("args", [(0.001, -0.01, 0.012), (0.005, 0.001, 0.009), (-0.05, -0.08, -0.02),
                                  (0.05, 0.01, 0.09), (0.0, -0.03, 0.03), (0.0, float("nan"), 0.1),
                                  (-0.015, -0.025, -0.005), (0.02, 0.0, 0.02)])
def test_decide(args):
    assert lab.decide(*args, margin=0.02) == U.decide(*args, margin=0.02)


def test_claim_problems():
    dec = {"r": "inconclusive", "e": "equivalent", "l": "a_lower"}
    claims = [{"id": "ok", "label": "MEASURED", "comparison": "e", "direction": "no-difference-detected"},
              {"id": "a", "label": "MEASURED", "comparison": "r", "direction": "higher", "scope": "frontier"},
              {"id": "b", "label": "MEASURED", "comparison": "l", "direction": "lower", "scope": "frontier"},
              {"id": "c", "label": "measured", "comparison": "zz"},
              {"id": "d", "label": "PUBLICLY DOCUMENTED", "scope": "frontier"},
              {"id": "e", "label": "MEASURED", "comparison": "zz", "direction": "lower"},
              {"id": "f", "label": "INFERENCE/SPECULATION", "scope": "frontier"}]
    want = [("CLAIM_DIRECTION", "a"), ("CLAIM_SCOPE", "a"), ("CLAIM_SCOPE", "b"), ("CLAIM_LABEL", "c"),
            ("CLAIM_SOURCE", "d"), ("CLAIM_COMPARISON", "e")]
    assert lab.claim_problems(claims, dec) == want


def test_claim_problems_on_the_sample(tmp_path):
    for flawed in (True, False):
        P = PK.load(SA.write_sample(tmp_path / str(flawed), flawed=flawed))
        dec = PK.recomputed_decisions(P["claim"], P["results"])
        ref = [(p.code, p.where.split("#")[1]) for p in PK.claim_problems(P["claims"], dec, "## Limits\n")]
        assert lab.claim_problems(P["claims"], dec) == ref


def test_comparison_problems_on_the_sample(tmp_path):
    import yaml
    for flawed in (True, False):
        pkg = SA.write_sample(tmp_path / f"c{flawed}", flawed=flawed)
        claim = yaml.safe_load((pkg / "claim.yaml").read_text())
        cards = {p.parent.name: yaml.safe_load(p.read_text()) for p in (pkg / "runs").glob("*/run_card.yaml")}
        got = [x for c in claim["comparisons"] for x in lab.comparison_problems(claim, c, cards)]
        ref = PK.parity_problems(pkg, claim)
        assert [x[0] for x in got] == [p.code for p in ref]
    assert got == []


def test_undeclared_change_is_parity_problem(tmp_path):
    import yaml
    pkg = SA.write_sample(tmp_path / "u", flawed=False)
    claim = yaml.safe_load((pkg / "claim.yaml").read_text())
    cards = {p.parent.name: yaml.safe_load(p.read_text()) for p in (pkg / "runs").glob("*/run_card.yaml")}
    cards["gspo-s1"]["args"]["temperature"] = 0.7
    comp = next(c for c in claim["comparisons"] if c["name"] == "reproduction")
    assert lab.comparison_problems(claim, comp, cards) == [("PARITY", "seed 1")]
    comp = dict(comp, changed=comp["changed"] + ["args.temperature"])
    assert lab.comparison_problems(claim, comp, cards) == []
