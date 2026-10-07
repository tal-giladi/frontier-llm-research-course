"""Tests for frontierlab.capstone (Module 20): uncertainty, claim registry, package checker, review, scaffold."""

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from frontierlab.capstone import claims as CL
from frontierlab.capstone import package as PK
from frontierlab.capstone import review as RV
from frontierlab.capstone import sample as SA
from frontierlab.capstone import uncertainty as U

TEMPLATE = Path(__file__).resolve().parents[3] / "templates" / "experiment-contract.md"


# ----------------------------------------------------------------------------------------- uncertainty

def test_hierarchical_bootstrap_basic():
    rng = np.random.default_rng(0)
    b = rng.normal(3.0, 0.1, size=(3, 200))
    a = b + 0.05 + rng.normal(0, 0.01, size=(3, 200))
    r = U.hierarchical_bootstrap(a, b)
    assert math.isclose(r["mean_diff"], float((a - b).mean()), rel_tol=1e-12)
    lo, hi = r["ci"]
    assert lo < r["mean_diff"] < hi and lo > 0.04 and hi < 0.06
    assert r["n_seeds"] == 3 and r["n_items"] == 200 and "hierarchical" in r["method"]
    with pytest.raises(ValueError):
        U.hierarchical_bootstrap(a[:, :10], b)


def test_hierarchical_bootstrap_sees_seed_variance():
    """A difference that varies only between seeds: the interval must be wider than an items-only bootstrap's."""
    W = 300
    d_seed = np.array([0.00, 0.04, 0.08])
    b = np.zeros((3, W))
    a = b + d_seed[:, None] + np.random.default_rng(1).normal(0, 0.001, size=(3, W))
    lo, hi = U.hierarchical_bootstrap(a, b)["ci"]
    assert hi - lo > 0.04                                   # items alone would give about 0.0003


def test_seed_t_interval_matches_reference():
    from frontierlab.posttrain.arms import t_interval
    a, b = [0.35, 0.31, 0.40], [0.30, 0.30, 0.33]
    r = U.seed_t_interval(a, b)
    m, lo, hi = t_interval([x - y for x, y in zip(a, b)])
    assert math.isclose(r["mean_diff"], m) and math.isclose(r["ci"][0], lo) and math.isclose(r["ci"][1], hi)
    assert all(math.isnan(x) for x in U.seed_t_interval([1.0], [0.5])["ci"])


@pytest.mark.parametrize("args,want", [((0.001, -0.01, 0.012), "equivalent"), ((0.005, 0.001, 0.009), "equivalent"),
                                       ((-0.05, -0.08, -0.02), "a_lower"), ((0.05, 0.01, 0.09), "a_higher"),
                                       ((0.0, -0.03, 0.03), "inconclusive"), ((0.0, float("nan"), 0.1), "inconclusive"),
                                       ((-0.015, -0.025, -0.005), "a_lower")])
def test_decide(args, want):
    assert U.decide(*args, margin=0.02) == want


def test_noise_floor():
    nf = U.noise_floor([0.402, 0.371, 0.388])
    sd = float(np.std([0.402, 0.371, 0.388], ddof=1))
    assert math.isclose(nf["seed_std"], sd) and math.isclose(nf["mde"], 2.8 * sd * math.sqrt(2 / 3))


# ----------------------------------------------------------------------------------------- claims

def test_claim_registry_complete():
    assert set(CL.CLAIMS) == {"gspo", "qkclip", "dsa", "mhc", "microanneal", "opd"}
    for c in CL.CLAIMS.values():
        assert c.url.startswith("https://") and c.where and c.statement and c.null_result and c.extension
        assert "PROJECTED" in c.main_path and c.main_gpu_hours[0] <= c.main_gpu_hours[1]
        assert len(c.questions) >= 3 and c.builds_on and c.code
        assert all(re.fullmatch(r"\d\d\.\d", x) for x in c.builds_on)


def test_projected_hours_qkclip():
    rung = lambda tok, fpt, mfu: CL.projected_gpu_hours(9, tok, fpt, mfu, overhead=0.03)   # noqa: E731
    assert abs(rung(2.62e8, 3.209e8, 0.2) - 1.09) < 0.01
    assert abs(rung(2.62e8, 6.136e8, 0.2) - 2.09) < 0.01
    assert abs(rung(2.49e9, 7.881e8, 0.3) - 17.0) < 0.05


def test_projected_hours_formula_uses_course_accounting():
    from frontierlab.attention.accounting import flops_per_token
    from frontierlab.model import PRESETS
    assert abs(flops_per_token(PRESETS["pilot-30m"](32768), 1024) - 3.209e8) / 3.209e8 < 1e-3
    assert abs(flops_per_token(PRESETS["baseline0"](32768), 1024) - 7.881e8) / 7.881e8 < 1e-3


# ----------------------------------------------------------------------------------------- package checker

def test_template_contract_is_not_filled():
    probs = PK.contract_problems(TEMPLATE.read_text(encoding="utf-8"), "tokens")
    codes = {p.code for p in probs}
    assert "CONTRACT_EMPTY" in codes and "CONTRACT_UNCHECKED" in codes


def test_scaffold_contract_is_filled():
    from frontierlab.capstone.scaffold import write_contract
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        write_contract(Path(d), "cpu", [0, 1, 2], None)
        text = (Path(d) / "contract.md").read_text(encoding="utf-8")
    assert PK.contract_problems(text, "tokens") == []
    assert [p.code for p in PK.contract_problems(text, "flops")] == ["CONTRACT_AXIS"]
    assert {p.code for p in PK.contract_problems(text.replace("- [x] causal", "- [ ] causal"), "tokens")} == {"CONTRACT_UNCHECKED"}


def test_clean_sample_is_sound(tmp_path):
    assert PK.check_package(SA.write_sample(tmp_path / "clean", flawed=False)) == []


def test_flawed_sample_finds_every_planted_weakness(tmp_path):
    probs = PK.check_package(SA.write_sample(tmp_path / "flawed", flawed=True))
    assert {p.code for p in probs} == set(SA.PLANTED)


def test_claims_within_evidence():
    dec = {"r": "inconclusive", "e": "equivalent", "l": "a_lower"}
    ok = [{"id": 1, "label": "MEASURED", "comparison": "r", "direction": "no-difference-detected", "scope": "course"},
          {"id": 2, "label": "MEASURED", "comparison": "e", "direction": "equivalent"},
          {"id": 3, "label": "MEASURED", "comparison": "l", "direction": "lower"},
          {"id": 4, "label": "PUBLICLY DOCUMENTED", "source": "paper sec 2", "scope": "frontier"},
          {"id": 5, "label": "INFERENCE/SPECULATION", "scope": "frontier"}]
    assert PK.claim_problems(ok, dec, "# r\n## Limits\n") == []
    bad = [{"id": "a", "label": "MEASURED", "comparison": "r", "direction": "higher"},
           {"id": "b", "label": "MEASURED", "comparison": "l", "direction": "lower", "scope": "frontier"},
           {"id": "c", "label": "measured"}, {"id": "d", "label": "PUBLICLY DOCUMENTED"},
           {"id": "e", "label": "MEASURED", "comparison": "zz", "direction": "lower"}]
    codes = [p.code for p in PK.claim_problems(bad, dec, "This proves it.")]
    assert codes == ["CLAIM_DIRECTION", "CLAIM_SCOPE", "CLAIM_LABEL", "CLAIM_SOURCE", "CLAIM_COMPARISON",
                     "REPORT_OVERCLAIM", "REPORT_LIMITS"]


def test_hand_edited_numbers_are_caught(tmp_path):
    pkg = SA.write_sample(tmp_path / "p", flawed=False)
    res = json.loads((pkg / "results.json").read_text())
    res["comparisons"][0]["mean_diff"] += 0.01
    res["comparisons"][0]["ci"] = [res["comparisons"][0]["ci"][0], res["comparisons"][0]["mean_diff"] + 0.01]
    del res["noise_floor"]
    (pkg / "results.json").write_text(json.dumps(res))
    assert {p.code for p in PK.check_package(pkg)} >= {"UNC_MEAN", "UNC_NOISE"}


# ----------------------------------------------------------------------------------------- review

def test_revision_kind_covers_every_checker_code():
    src = (Path(PK.__file__)).read_text(encoding="utf-8")
    codes = set(re.findall(r'Problem\("([A-Z_]+)"', src))
    assert codes <= set(RV.REVISION_KIND) and set(RV.REVISION_KIND.values()) <= set(RV.KINDS)
    assert set(RV.BLOCKING) <= set(RV.REVISION_KIND)


def test_questions(tmp_path):
    pkg = SA.write_sample(tmp_path / "f", flawed=True)
    qs = RV.questions(pkg)
    ids = [q["id"] for q in qs]
    assert len(ids) == len(set(ids))
    assert sum(q["kind"] == "generic" for q in qs) == 4 and sum(q["kind"] == "claim" for q in qs) == 3
    assert sum(q["kind"] == "problem" for q in qs) == len(PK.check_package(pkg))


def test_defence_problems():
    qs = [{"id": "G1"}, {"id": "P2"}, {"id": "P3"}]
    text = ("# Defence\n\n### G1 axis\nEqual tokens, because the claim is about loss per token. "
            "See contract.md for what it does not answer.\n\n### P2\nN/A\n")
    codes = [(p.code, p.where) for p in RV.defence_problems(text, qs)]
    assert codes == [("DEF_EMPTY", "P2"), ("DEF_EVIDENCE", "P2"), ("DEF_MISSING", "P3")]


def test_full_revision_passes(tmp_path):
    pkg = SA.write_sample(tmp_path / "rev", flawed=True)
    before = PK.check_package(pkg)
    new = SA.apply_reruns(pkg)
    assert all((pkg / f).exists() for f in new)
    SA.revise_by_hand(pkg)
    after = PK.check_package(pkg)
    assert after == []
    log = SA.reference_log(before)
    contract = (pkg / "contract.md").read_text(encoding="utf-8")
    assert RV.revision_log_problems(log, before, after, pkg, contract) == []
    # declining a blocking problem, the wrong kind, a missing entry and an unrecorded change are all caught
    bad = [dict(e) for e in log]
    bad[0]["kind"] = "rewrite"                                       # CONTRACT_STATUS needs "declare"
    i = next(i for i, e in enumerate(bad) if e["problem"].startswith("PARITY"))
    bad[i] = {"problem": bad[i]["problem"], "kind": "decline", "reason": "no time"}
    bad = [e for e in bad if not e["problem"].startswith("REPORT_OVERCLAIM")]
    codes = {p.code for p in RV.revision_log_problems(bad, before, after, pkg, contract.split("## Changes")[0] +
                                                           "## Changes after results\n\nNone.\n")}
    assert codes == {"REV_KIND", "REV_DECLINE", "REV_UNLOGGED", "REV_CONTRACT"}


# ----------------------------------------------------------------------------------------- scaffold (end to end)

def test_scaffold_correctness_checks():
    from frontierlab.capstone.scaffold import correctness_checks
    c = correctness_checks()
    assert c["passed"] and c["qkclip_cap_max_abs_err"] < 1e-9 and c["qkclip_refuses_qk_norm"]


def test_scaffold_end_to_end_short(tiny_data, tmp_path):
    """The whole pipeline at 6 steps on synthetic data: run cards, parents, parity, results, a sound package."""
    from frontierlab.capstone import scaffold as SC
    res = SC.run(tmp_path / "cap", seeds=[0, 1], steps=6, data=str(tiny_data), eval_n=16)
    pkg = tmp_path / "cap"
    assert res["problems"] == [], res["problems"]
    cards = {p.parent.name: yaml.safe_load(p.read_text()) for p in (pkg / "runs").glob("*/run_card.yaml")}
    assert len(cards) == 6
    assert cards["muon-clip-s1"]["parent_run"] == "muon-noqk-s1" and cards["muon-noqk-s0"]["parent_run"] == SC.EXTERNAL_PARENT
    assert {c["name"] for c in res["comparisons"]} == {"reproduction", "extension"}
    for c in res["comparisons"]:
        assert c["ci"][0] <= c["mean_diff"] <= c["ci"][1] and c["decision"] in U.DECISIONS
    assert res["tau"] >= 1 and res["secondary"]["muon-qknorm-s0"]["clipped_head_updates"] == 0
    # an undeclared second change is caught: give one clip run a different learning rate in its card
    card = cards["muon-clip-s0"]
    card["args"]["lr"] = 0.02
    (pkg / "runs" / "muon-clip-s0" / "run_card.yaml").write_text(yaml.safe_dump(card))
    assert any(p.code == "PARITY" and "args.lr" in p.message for p in PK.check_package(pkg))
