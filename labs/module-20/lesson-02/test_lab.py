import pytest

from frontierlab.capstone import package as PK
from frontierlab.capstone import review as RV
from frontierlab.capstone import sample as SA
from frontierlab.labkit import load_target

lab = load_target(__file__)


def pairs(problems):
    return [(p.code, p.where) for p in problems]


def test_revision_kind_matches_reference():
    for code, kind in RV.REVISION_KIND.items():
        assert lab.revision_kind(code) == kind, code
    with pytest.raises(KeyError):
        lab.revision_kind("NOT_A_CODE")


def test_defence_problems_small():
    qs = [{"id": "G1"}, {"id": "P2"}, {"id": "P3"}, {"id": "C1"}]
    text = ("# Defence\n\n### G1 axis\n> Why this axis?\nEqual tokens, because the claim is about loss per token. "
            "See contract.md for what it does not answer.\n\n### P2\nN/A\n\n### C1\nWe chose tau by a rule. It "
            "was fixed before the runs.\n")
    assert lab.defence_problems(text, qs) == pairs(RV.defence_problems(text, qs))
    assert lab.defence_problems(text, qs) == [("DEF_EMPTY", "P2"), ("DEF_EVIDENCE", "P2"), ("DEF_MISSING", "P3"),
                                              ("DEF_EVIDENCE", "C1")]


def test_defence_problems_on_the_sample(tmp_path):
    pkg = SA.write_sample(tmp_path / "s", flawed=True)
    qs = RV.questions(pkg)
    good = SA.reference_defence(qs)
    assert lab.defence_problems(good, qs) == []
    bad = good.replace("### G3", "### G9").replace("results.json, noise_floor", "the noise floor")
    assert lab.defence_problems(bad, qs) == pairs(RV.defence_problems(bad, qs)) != []


def test_revision_log_on_the_sample(tmp_path):
    pkg = SA.write_sample(tmp_path / "r", flawed=True)
    before = PK.check_package(pkg)
    SA.apply_reruns(pkg)
    middle = PK.check_package(pkg)                       # reruns done, nothing rewritten yet
    log = SA.reference_log(before)
    contract = (pkg / "contract.md").read_text(encoding="utf-8")
    assert lab.revision_log_problems(log, before, middle, pkg, contract) == \
        pairs(RV.revision_log_problems(log, before, middle, pkg, contract))
    SA.revise_by_hand(pkg)
    after = PK.check_package(pkg)
    contract = (pkg / "contract.md").read_text(encoding="utf-8")
    assert lab.revision_log_problems(log, before, after, pkg, contract) == []
    bad = [dict(e) for e in log]
    bad[0]["kind"] = "rewrite"
    i = next(i for i, e in enumerate(bad) if e["problem"].startswith("PARITY"))
    bad[i] = {"problem": bad[i]["problem"], "kind": "decline", "reason": "out of time"}
    j = next(i for i, e in enumerate(bad) if e["problem"].startswith("REPORT_LIMITS") or e["kind"] == "rewrite"
             and e["problem"].startswith("REPORT"))
    bad.pop(j)
    k = next(i for i, e in enumerate(bad) if e.get("files") and e["problem"].startswith("SEEDS"))
    bad[k] = dict(bad[k], files=["runs/gspo-stale4-s7"])
    bad.append({"problem": "UNC_MEAN@results.json#control", "kind": "reanalyse", "change": "x"})
    none = contract.split("## Changes")[0] + "## Changes after results\n\nNone yet.\n"
    want = pairs(RV.revision_log_problems(bad, before, after, pkg, none))
    assert {c for c, _ in want} == {"REV_KIND", "REV_DECLINE", "REV_UNLOGGED", "REV_FILES", "REV_CONTRACT", "REV_UNKNOWN"}
    assert lab.revision_log_problems(bad, before, after, pkg, none) == want
