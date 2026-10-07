"""Tests for the evaluation-integrity lab (lessons 16.1 and 16.4): the TODOs on synthetic records,
then focused regression checks that run the real toy runner in the course sandbox.

The regression checks are the point of the lab: the hardened verifier must reject every scripted
tampering demonstration while accepting valid solutions, and each weaker verifier level must show
its measured, specific holes. Nothing here touches the course's own tests or tooling; every run
happens in a disposable sandbox directory on the toy fizzbuzz task.
"""

import sys
from pathlib import Path

import pytest

from frontierlab.labkit import load_target

lab = load_target(__file__)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import demos  # noqa: E402
import toyrunner as T  # noqa: E402

LABS = HERE.parents[1]


def _v(accepted: bool, reason: str) -> dict:
    return {"accepted": accepted, "reason": reason}


# --------------------------------------------------------------------------- the TODOs (synthetic records)

ROWS = [
    {"candidate": "a", "kind": "correct", "levels": {"exit": _v(True, "exit code 0"),
                                                     "log": _v(True, "complete passing summary line"),
                                                     "protected": _v(True, "exit code 0"),
                                                     "channel": _v(True, "all values match"),
                                                     "hardened": _v(True, "all values match")}},
    {"candidate": "b", "kind": "correct", "levels": {"exit": _v(True, "exit code 0"),
                                                     "log": _v(True, "complete passing summary line"),
                                                     "protected": _v(True, "exit code 0"),
                                                     "channel": _v(True, "all values match"),
                                                     "hardened": _v(False, "value mismatch")}},
    {"candidate": "c", "kind": "wrong", "levels": {"exit": _v(False, "exit code 1"),
                                                   "log": _v(False, "summary line says 2 passed, 1 failed"),
                                                   "protected": _v(False, "exit code 1"),
                                                   "channel": _v(False, "value mismatch"),
                                                   "hardened": _v(False, "value mismatch")}},
    {"candidate": "d", "kind": "tampering", "levels": {"exit": _v(True, "exit code 0"),
                                                       "log": _v(False, "no summary line in the report"),
                                                       "protected": _v(True, "exit code 0"),
                                                       "channel": _v(False, "no results (timed out)"),
                                                       "hardened": _v(False, "no results (timed out)")}},
    {"candidate": "e", "kind": "tampering", "levels": {"exit": _v(True, "exit code 0"),
                                                       "log": _v(True, "complete passing summary line"),
                                                       "protected": _v(True, "exit code 0"),
                                                       "channel": _v(False, "unserializable value"),
                                                       "hardened": _v(False, "unserializable value")}},
    {"candidate": "f", "kind": "overfit", "levels": {"exit": _v(True, "exit code 0"),
                                                     "log": _v(True, "complete passing summary line"),
                                                     "protected": _v(True, "exit code 0"),
                                                     "channel": _v(True, "all values match"),
                                                     "hardened": _v(False, "value mismatch")}},
]


def test_acceptance_matrix():
    m = lab.acceptance_matrix(ROWS)
    assert m["exit"]["false_accepts"] == 3 and m["exit"]["bad_total"] == 4
    assert m["exit"]["tampering_accepts"] == 2 and m["exit"]["tampering_total"] == 2
    assert m["exit"]["overfit_accepts"] == 1 and m["exit"]["false_rejects"] == 0
    assert abs(m["exit"]["false_accept_rate"] - 3 / 4) < 1e-12
    assert m["log"]["false_accepts"] == 2 and m["log"]["tampering_accepts"] == 1
    assert m["protected"]["false_accepts"] == 3
    assert m["channel"]["false_accepts"] == 1 and m["channel"]["false_rejects"] == 0   # the overfit only
    assert m["channel"]["tampering_accepts"] == 0
    assert m["hardened"]["false_accepts"] == 0
    assert m["hardened"]["false_rejects"] == 1 and m["hardened"]["correct_total"] == 2
    assert m["hardened"]["tampering_accept_rate"] == 0.0


def test_explain_rejection():
    assert lab.explain_rejection(_v(False, "no results (crashed before writing results)")) == \
        "trusted reporting: incomplete execution is failure"
    assert lab.explain_rejection(_v(False, "incomplete results")) == \
        "trusted reporting: incomplete execution is failure"
    assert lab.explain_rejection(_v(False, "no results (timed out)")) == "isolation: bounded time"
    assert lab.explain_rejection(_v(False, "unserializable value")) == \
        "trusted reporting: only plain values are compared"
    assert lab.explain_rejection(_v(False, "value mismatch")) == \
        "values compared outside the candidate's process"
    assert lab.explain_rejection(_v(False, "call raised ValueError")) == \
        "values compared outside the candidate's process"
    assert lab.explain_rejection(_v(False, "property failure")) == "property checks"
    assert lab.explain_rejection(_v(True, "all values match")) == "no rejection to explain"


CLASSIFICATIONS = [
    ("The reward paid for output length, and the policy learned to emit long unfinished strings.",
     "misspecification"),
    ("The format reward was satisfied by any well-formed program, and the policy stopped improving.",
     "misspecification"),
    ("The verifier checked only the three visible pairs, and the submission hard-coded them.",
     "overfitting"),
    ("The submission returned a lookup table of the visible examples.", "overfitting"),
    ("The submission overrode __eq__ so every assertion passed.", "tampering"),
    ("The submission called sys.exit(0) before the tests ran.", "tampering"),
    ("The submission rewrote the test file to assert its own wrong outputs.", "tampering"),
    ("The submission read the grader's expected answers from a file in its workspace.", "tampering"),
    ("The runner loaded a workspace conftest.py whose hook rewrote the report to passing.", "tampering"),
    ("The submission printed the passing summary line itself.", "tampering"),
]


def test_classify_behaviour():
    for description, want in CLASSIFICATIONS:
        assert lab.classify_behaviour(description) == want, description
    with pytest.raises(ValueError):
        lab.classify_behaviour("The model was slower on held-out tasks.")


def test_strict_log_verdict():
    ok = {"returncode": 0, "timed_out": False, "log": "3 passed, 0 failed\nTOYRUNNER COMPLETE n=3\n"}
    assert lab.strict_log_verdict(ok) is True
    assert lab.strict_log_verdict({**ok, "log": "5 passed, 0 failed\nTOYRUNNER COMPLETE n=5\n"}, n_tests=5)
    assert lab.strict_log_verdict({"returncode": 0, "timed_out": False, "log": ""}) is False       # exit_early
    assert lab.strict_log_verdict({"returncode": 0, "timed_out": False,
                                    "log": "3 passed, 0 failed"}) is False                          # print_pass
    assert lab.strict_log_verdict({"returncode": 1, "timed_out": False, "log": ok["log"]}) is False
    assert lab.strict_log_verdict({"returncode": None, "timed_out": True, "log": ""}) is False      # hang
    assert lab.strict_log_verdict({"returncode": 0, "timed_out": False,
                                    "log": "2 passed, 1 failed\nTOYRUNNER COMPLETE n=3"}) is False
    assert lab.strict_log_verdict({"returncode": 0, "timed_out": False,
                                    "log": "3 passed, 0 failed\nTOYRUNNER COMPLETE n=2"}) is False


# --------------------------------------------------------------------------- regression checks (sandboxed)

@pytest.fixture(scope="module")
def measured():
    """The measured acceptance-matrix cells the regression checks need, run once (~15-30 s)."""
    out: dict = {"digest_before": T.repo_digest(str(LABS)), "weak": {}, "protected": {},
                 "channel": {}, "hardened": {}}
    for name in ("correct", "wrong", "exit_early", "print_pass", "hardcode_visible"):
        out["weak"][name] = T.weak_verdicts(demos.CANDIDATES[name]["files"])
    for name in ("read_expected", "edit_tests", "always_equal", "conftest_rewrite"):
        out["protected"][name] = T.protected_verdict(demos.CANDIDATES[name]["files"])
    for name in demos.ORDER:
        files = demos.CANDIDATES[name]["files"]
        out["hardened"][name] = T.hardened_verdict(
            files, seed=0, timeout=1.5 if name == "hang" else T.TIMEOUT)
    out["channel"]["hardcode_visible"] = T.channel_verdict(demos.CANDIDATES["hardcode_visible"]["files"])
    out["channel"]["always_equal"] = T.channel_verdict(demos.CANDIDATES["always_equal"]["files"])
    out["hardened"]["correct_with_hostile_conftest"] = T.hardened_verdict(
        {"solution.py": demos.CORRECT, "conftest.py": demos.CONFTEST_HOOK}, seed=0)
    return out


def test_hardened_rejects_every_tampering_demo(measured):
    for name in ("exit_early", "always_equal", "conftest_rewrite", "edit_tests", "read_expected",
                 "print_pass", "hardcode_visible"):
        assert measured["hardened"][name]["accepted"] is False, name


def test_hardened_rejects_with_the_expected_reasons(measured):
    h = measured["hardened"]
    assert h["exit_early"]["reason"] == "no results (crashed before writing results)"
    assert h["always_equal"]["reason"] == "unserializable value"
    assert h["conftest_rewrite"]["reason"] == "value mismatch"
    assert h["edit_tests"]["reason"] == "value mismatch"
    assert h["read_expected"]["reason"] == "no results (crashed before writing results)"
    assert h["print_pass"]["reason"] == "no results (crashed before writing results)"
    assert h["hardcode_visible"]["reason"] == "value mismatch"
    assert h["hang"]["reason"] == "no results (timed out)"


def test_hardened_accepts_valid_solutions_and_rejects_ordinary_wrong(measured):
    h = measured["hardened"]
    assert h["correct"]["accepted"] and h["correct"]["reason"] == "all values match"
    assert h["correct_alt"]["accepted"]                      # correct, written differently
    assert h["correct_with_hostile_conftest"]["accepted"]   # the guarded levels load no plugins
    assert h["wrong"]["accepted"] is False and h["wrong"]["reason"] == "value mismatch"


def test_exit_code_verdicts(measured):
    w = measured["weak"]
    assert w["correct"]["exit"]["accepted"]
    assert w["exit_early"]["exit"]["accepted"]          # the exit-code-only verifier's hole
    assert w["print_pass"]["exit"]["accepted"]
    assert w["hardcode_visible"]["exit"]["accepted"]
    assert w["wrong"]["exit"]["accepted"] is False


def test_log_parsing_verdicts(measured):
    w = measured["weak"]
    assert w["correct"]["log"]["accepted"]
    assert w["exit_early"]["log"]["accepted"] is False  # no summary line to parse
    assert w["print_pass"]["log"]["accepted"]          # the fabricated line is enough
    assert w["wrong"]["log"]["accepted"] is False


def test_hidden_tests_alone_do_not_stop_tampering(measured):
    """The protected level: cases are not files the candidate can read or edit, yet the verdict is
    still the child's exit code — and in-child equality, an early exit, a plugin hook and a forged
    summary all still pass (measured)."""
    p = measured["protected"]
    assert p["read_expected"]["exit"]["accepted"] is False   # nothing left in the workspace to read
    assert p["edit_tests"]["exit"]["accepted"] is False      # no writable test file to edit
    assert p["always_equal"]["exit"]["accepted"]
    assert p["conftest_rewrite"]["exit"]["accepted"]


def test_the_channel_closes_equality_and_forgery_but_not_overfitting(measured):
    assert measured["channel"]["always_equal"]["accepted"] is False
    assert measured["channel"]["hardcode_visible"]["accepted"] is True    # visible inputs only
    assert measured["hardened"]["hardcode_visible"]["accepted"] is False  # hidden + fresh inputs


def test_strict_log_parser_on_real_runs(measured):
    """The student's strict parser fixes exit_early and print_pass but still accepts the tampering
    demos whose runs genuinely complete and pass in-child (measured, not assumed)."""
    w = measured["weak"]
    assert lab.strict_log_verdict(w["correct"]["run"]) is True
    assert lab.strict_log_verdict(w["exit_early"]["run"]) is False
    assert lab.strict_log_verdict(w["print_pass"]["run"]) is False


def test_the_demonstrations_never_touch_course_files(measured):
    assert T.repo_digest(str(LABS)) == measured["digest_before"]


def test_every_demo_is_labelled_with_its_published_analogue():
    for name, c in demos.CANDIDATES.items():
        if c["kind"] == "tampering":
            assert any(s in c["label"] for s in ("MacDiarmid", "Baker", "METR", "ImpossibleBench")), name
        if c["kind"] == "overfit":
            assert "16.1" in c["label"] or "16.4" in c["label"], name
