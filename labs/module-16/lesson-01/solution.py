"""Reference solution for lab 16.1 — environments and verifiers."""

from __future__ import annotations

import math


def confusion(rows: list[dict]) -> dict:
    out = {}
    for v in sorted({r["verifier"] for r in rows}):
        rs = [r for r in rows if r["verifier"] == v]
        good = [r for r in rs if r["kind"] == "correct"]
        bad = [r for r in rs if r["kind"] != "correct"]
        loop = [r for r in bad if r["kind"] == "loophole"]
        out[v] = {"false_accept_rate": sum(r["passed"] for r in bad) / len(bad) if bad else float("nan"),
                  "false_reject_rate": sum(not r["passed"] for r in good) / len(good) if good else float("nan"),
                  "loophole_accept_rate": sum(r["passed"] for r in loop) / len(loop) if loop else float("nan")}
    return out


def miss_probability(p_fail: float, n: int) -> float:
    return (1.0 - p_fail) ** n


def tests_needed(p_fail: float, delta: float) -> int:
    if p_fail <= 0:
        return -1
    if p_fail >= 1:
        return 1
    return max(1, math.ceil(math.log(delta) / math.log(1.0 - p_fail)))


def reset_verdict(check: dict) -> str:
    if not check["disturb_changed_state"]:
        return "untested"
    if check["reset_restores"] and check["fresh_matches"]:
        return "clean"
    return "leaks"
