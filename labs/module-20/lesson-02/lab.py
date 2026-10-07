"""Lab 20.2 — review, defence and revision. Fill in the TODOs; run `pytest labs/module-20/lesson-02` to check, then
follow the steps in the lesson with `python labs/module-20/lesson-02/review_lab.py`.

Problems are ``frontierlab.capstone.package.Problem`` objects with ``code``, ``where``, ``message`` and ``key``
(``"CODE@where"``). Your functions return ``(code, where)`` pairs. Write them without importing the reference
versions from ``frontierlab.capstone.review``; the two helpers imported below are allowed.
"""

from __future__ import annotations

import re  # noqa: F401
from pathlib import Path  # noqa: F401

from frontierlab.capstone.review import EVIDENCE, defence_sections  # noqa: F401  (helpers you may use)

KINDS = ("rerun", "reanalyse", "rewrite", "declare")
# Problems that make the result not count until fixed: they cannot be declined.
BLOCKING = ("PARITY", "TUNING", "SEEDS", "UNC_SEEDS", "UNC_INTERVAL", "UNC_DECISION", "CLAIM_DIRECTION",
            "CLAIM_SCOPE", "CONTRACT_UNCHECKED", "RUNCARD_MISSING")


def revision_kind(code: str) -> str:
    """Which revision a checker problem needs, one of KINDS:

    * ``rerun`` — the evidence itself is not comparable or too thin, only new runs fix it: RUNCARD_MISSING, PARITY,
      TUNING, SEEDS, UNC_SEEDS;
    * ``reanalyse`` — the runs are fine, the numbers drawn from them are not: RESULTS_MISSING and every other UNC_ code;
    * ``rewrite`` — the evidence is fine, the words claim more than it supports: every CLAIM_ and REPORT_ code;
    * ``declare`` — the record hides or omits something that must be stated: every CONTRACT_ and PKG_ code, and
      RUNCARD_PARENT.

    Raise KeyError for any other code.
    """
    raise NotImplementedError("TODO 1: map a problem code to the revision it needs")


def defence_problems(text: str, questions: list[dict]) -> list[tuple[str, str]]:
    """Check a written defence against the reviewer questions (``{"id": "G1", ...}``), in question order.

    For each question: ("DEF_MISSING", id) if there is no ``### <id>`` section (use ``defence_sections``); otherwise
    ("DEF_EMPTY", id) if the answer has fewer than two sentence ends (``.``, ``!`` or ``?`` followed by whitespace or
    the end) or is just "N/A", "none" or "TBD"; then ("DEF_EVIDENCE", id) if ``EVIDENCE.search(answer)`` finds no
    package file and no interval.
    """
    raise NotImplementedError("TODO 2: check the defence")


def revision_log_problems(log: list[dict], before: list, after: list, pkg=None,
                          contract_text: str | None = None) -> list[tuple[str, str]]:
    """Check a revision log: entries ``{problem: "CODE@where", kind, change, files, result_changed, reason}``.

    For each problem in ``before`` (in order), with its entry looked up by ``problem.key``:

    * no entry: ("REV_UNLOGGED", key);
    * kind "decline": ("REV_DECLINE", key) if the code is BLOCKING or the reason has fewer than 8 words; nothing else;
    * otherwise: ("REV_KIND", key) if kind != revision_kind(code); ("REV_NOT_FIXED", key) if the key is still among
      ``after``'s keys; for kind "rerun", ("REV_FILES", key) if ``files`` is empty or (with ``pkg``) a file does not
      exist under ``pkg``; if ``result_changed``, ("REV_CONTRACT", key) when ``contract_text`` is None or the text
      after "## Changes after results", stripped, lower-cased and without a final ".", is "", "none" or "none yet".

    Finally ("REV_UNKNOWN", key) for every log entry whose problem was not in ``before``.
    """
    raise NotImplementedError("TODO 3: check the revision log")
