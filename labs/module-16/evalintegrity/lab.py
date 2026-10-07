"""Lab (evaluation integrity, lessons 16.1 and 16.4) — reading a verifier attack honestly.

Fill in the TODOs; `pytest labs/module-16/evalintegrity` checks them. `tamper_lab.py` uses these
functions on the measured acceptance matrix of the toy runner (`labs/module-16/evalintegrity/
toyrunner.py`), whose five verifier levels form a defence ladder from "trust the exit code" to the
hardened verifier of lesson 16.1.

Trace records look like this (one per candidate, see `traces/acceptance-matrix.jsonl`):

    {"candidate": "exit_early", "kind": "tampering",
     "levels": {"exit": {"accepted": true,  "reason": "exit code 0"},
                "log":  {"accepted": false, "reason": "no summary line in the report"},
                ...
                "hardened": {"accepted": false, "reason": "no results (crashed before writing results)"}}}
"""

from __future__ import annotations

import re

# The five levels of the toy runner's defence ladder, weakest to strongest (toyrunner.LEVELS).
LEVELS = ("exit", "log", "protected", "channel", "hardened")


def acceptance_matrix(rows: list[dict], levels: tuple = LEVELS) -> dict:
    """Per verifier level, from the trace records of ``toyrunner.run_candidate``:

    - "false_accepts" / "bad_total": accepted rows whose "kind" is "wrong", "tampering" or
      "overfit", over all such rows. A false accept is reward flowing to behaviour nobody wanted.
    - "tampering_accepts" / "tampering_total": the same for "tampering" rows only.
    - "overfit_accepts": accepted "overfit" rows (total 1 in this lab).
    - "false_rejects" / "correct_total": rejected rows of kind "correct", over all correct rows.
    - the corresponding rates ("false_accept_rate", "tampering_accept_rate", "false_reject_rate"),
      NaN where the total is 0.

    A verifier is usable as a reward if false_accepts is 0 on the registered candidates and
    false_rejects is 0; say the totals when you report a rate.
    """
    raise NotImplementedError("TODO 1: the acceptance matrix per verifier level")


def explain_rejection(verdict: dict) -> str:
    """Name the defence that produced one rejected verdict {"accepted": False, "reason": ...} of the
    hardened level. Return exactly one of:

    - "trusted reporting: incomplete execution is failure" (the child died, hung or wrote no
      complete results: reasons containing "no results" or "incomplete"),
    - "isolation: bounded time" (the reason contains "timed out"),
    - "trusted reporting: only plain values are compared" ("unserializable"),
    - "values compared outside the candidate's process" ("mismatch" or "call raised"),
    - "property checks" ("property"),
    - and "no rejection to explain" for an accepted verdict.
    """
    raise NotImplementedError("TODO 2: which defence caught it")


def classify_behaviour(description: str) -> str:
    """Classify how a submission earned its reward: "tampering" (it changed what the harness reads —
    its tests, its report, its answers, its equality, its process), "overfitting" (it special-cased
    the inputs the reward checks, without touching the harness) or "misspecification" (the reward
    paid for something else entirely, such as form or length). The descriptions this must handle:

    - "The reward paid for output length, and the policy learned to emit long unfinished strings."
    - "The format reward was satisfied by any well-formed program, and the policy stopped improving."
    - "The verifier checked only the three visible pairs, and the submission hard-coded them."
    - "The submission returned a lookup table of the visible examples."
    - "The submission overrode __eq__ so every assertion passed."
    - "The submission called sys.exit(0) before the tests ran."
    - "The submission rewrote the test file to assert its own wrong outputs."
    - "The submission read the grader's expected answers from a file in its workspace."
    - "The runner loaded a workspace conftest.py whose hook rewrote the report to passing."
    - "The submission printed the passing summary line itself."

    Raise ValueError for a description that matches none of the three modes.
    """
    raise NotImplementedError("TODO 3: tampering, overfitting or misspecification")


def strict_log_verdict(run: dict, n_tests: int = 3) -> bool:
    """Your hardened log parser (the "fix" exercise). ``run`` is a weak run of the toy runner:
    {"returncode": int | None, "timed_out": bool, "log": str}. Accept iff all of:

    - the run was not killed by the timeout and exited 0,
    - the log contains the runner's completeness marker "TOYRUNNER COMPLETE n=<n_tests>",
    - the log contains a summary line reporting exactly n_tests passed and 0 failed.

    The lab runs it on every candidate and compares with the exit-code and log-parsing verifiers:
    it rejects exit_early and print_pass, but every tampering demo that produces a genuinely
    complete passing run still passes — a report the child wrote is not evidence about the child,
    whatever you check about its shape.
    """
    raise NotImplementedError("TODO 4: the strict report parser")
