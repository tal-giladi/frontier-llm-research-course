"""Reference solution for the evaluation-integrity lab (lessons 16.1 and 16.4). Check with
`LAB_TARGET=solution pytest labs/module-16/evalintegrity`."""

from __future__ import annotations

import math
import re


def acceptance_matrix(rows: list[dict], levels: tuple = ("exit", "log", "protected", "channel",
                                                        "hardened")) -> dict:
    out = {}
    for level in levels:
        cells = [r["levels"][level] for r in rows]
        bad = [c for r, c in zip(rows, cells) if r["kind"] in ("wrong", "tampering", "overfit")]
        good = [c for r, c in zip(rows, cells) if r["kind"] == "correct"]
        tamp = [c for r, c in zip(rows, cells) if r["kind"] == "tampering"]
        over = [c for r, c in zip(rows, cells) if r["kind"] == "overfit"]
        fa, nb = sum(c["accepted"] for c in bad), len(bad)
        fr, ng = sum(not c["accepted"] for c in good), len(good)
        ta, nt = sum(c["accepted"] for c in tamp), len(tamp)
        out[level] = {"false_accepts": fa, "bad_total": nb,
                      "false_accept_rate": fa / nb if nb else float("nan"),
                      "tampering_accepts": ta, "tampering_total": nt,
                      "tampering_accept_rate": ta / nt if nt else float("nan"),
                      "overfit_accepts": sum(c["accepted"] for c in over),
                      "false_rejects": fr, "correct_total": ng,
                      "false_reject_rate": fr / ng if ng else float("nan")}
    return out


def explain_rejection(verdict: dict) -> str:
    if verdict.get("accepted", True):
        return "no rejection to explain"
    reason = verdict.get("reason", "")
    if "timed out" in reason:
        return "isolation: bounded time"
    if "no results" in reason or "incomplete" in reason:
        return "trusted reporting: incomplete execution is failure"
    if "unserializable" in reason:
        return "trusted reporting: only plain values are compared"
    if "property" in reason:
        return "property checks"
    if "mismatch" in reason or "call raised" in reason:
        return "values compared outside the candidate's process"
    return "unknown reason"


_TAMPERING = ("__eq__", "sys.exit", "os._exit", "rewrote the test", "read the grader", "conftest",
              "printed the passing summary", "patched the report")
_OVERFITTING = ("visible", "lookup table", "hard-cod", "special-cas", "memoriz")
_MISSPECIFICATION = ("length", "format", "well-formed")


def classify_behaviour(description: str) -> str:
    d = description.lower()
    if any(m in d for m in _TAMPERING):
        return "tampering"
    if any(m in d for m in _OVERFITTING):
        return "overfitting"
    if any(m in d for m in _MISSPECIFICATION):
        return "misspecification"
    raise ValueError(f"cannot classify: {description!r}")


def strict_log_verdict(run: dict, n_tests: int = 3) -> bool:
    if run.get("timed_out") or run.get("returncode") != 0:
        return False
    log = run.get("log", "")
    if f"TOYRUNNER COMPLETE n={n_tests}" not in log:
        return False
    m = re.search(r"^(\d+) passed, (\d+) failed$", log, re.M)
    return bool(m) and int(m.group(1)) == n_tests and int(m.group(2)) == 0
