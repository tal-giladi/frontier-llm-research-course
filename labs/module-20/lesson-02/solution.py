"""Lab 20.2 — reference solution. Check it with `LAB_TARGET=solution pytest labs/module-20/lesson-02`."""

from __future__ import annotations

import re
from pathlib import Path

from frontierlab.capstone.review import EVIDENCE, defence_sections

KINDS = ("rerun", "reanalyse", "rewrite", "declare")
BLOCKING = ("PARITY", "TUNING", "SEEDS", "UNC_SEEDS", "UNC_INTERVAL", "UNC_DECISION", "CLAIM_DIRECTION",
            "CLAIM_SCOPE", "CONTRACT_UNCHECKED", "RUNCARD_MISSING")
RERUN = ("RUNCARD_MISSING", "PARITY", "TUNING", "SEEDS", "UNC_SEEDS")
KNOWN = {"PKG_CLAIM", "PKG_ROLES", "PKG_ARMS", "CONTRACT_MISSING", "CONTRACT_SECTION", "CONTRACT_EMPTY",
         "CONTRACT_STATUS", "CONTRACT_AXIS", "CONTRACT_RULE", "CONTRACT_UNCHECKED", "RUNCARD_MISSING", "RUNCARD_PARENT",
         "PARITY", "TUNING", "SEEDS", "UNC_SEEDS", "RESULTS_MISSING", "UNC_INTERVAL", "UNC_MEAN", "UNC_DECISION",
         "UNC_NOISE", "CLAIM_LABEL", "CLAIM_SOURCE", "CLAIM_COMPARISON", "CLAIM_DIRECTION", "CLAIM_SCOPE",
         "REPORT_MISSING", "REPORT_OVERCLAIM", "REPORT_LIMITS"}


def revision_kind(code: str) -> str:
    if code not in KNOWN:
        raise KeyError(code)
    if code in RERUN:
        return "rerun"
    if code.startswith("UNC_") or code == "RESULTS_MISSING":
        return "reanalyse"
    if code.startswith(("CLAIM_", "REPORT_")):
        return "rewrite"
    return "declare"


def defence_problems(text: str, questions: list[dict]) -> list[tuple[str, str]]:
    secs, out = defence_sections(text), []
    for q in questions:
        body = secs.get(q["id"])
        if body is None:
            out.append(("DEF_MISSING", q["id"]))
            continue
        if len(re.findall(r"[.!?](\s|$)", body)) < 2 or re.fullmatch(r"(?i)\W*(n/?a|none|tbd)\W*", body):
            out.append(("DEF_EMPTY", q["id"]))
        if not EVIDENCE.search(body):
            out.append(("DEF_EVIDENCE", q["id"]))
    return out


def revision_log_problems(log: list[dict], before: list, after: list, pkg=None,
                          contract_text: str | None = None) -> list[tuple[str, str]]:
    out = []
    entries = {e.get("problem"): e for e in (log or [])}
    still = {p.key for p in after}
    for p in before:
        e = entries.get(p.key)
        if e is None:
            out.append(("REV_UNLOGGED", p.key))
            continue
        kind = e.get("kind")
        if kind == "decline":
            if p.code in BLOCKING or len((e.get("reason") or "").split()) < 8:
                out.append(("REV_DECLINE", p.key))
            continue
        if kind != revision_kind(p.code):
            out.append(("REV_KIND", p.key))
        if p.key in still:
            out.append(("REV_NOT_FIXED", p.key))
        if kind == "rerun":
            files = e.get("files") or []
            if not files or (pkg is not None and not all((Path(pkg) / f).exists() for f in files)):
                out.append(("REV_FILES", p.key))
        if e.get("result_changed"):
            tail = (contract_text or "").split("## Changes after results")[-1].strip().lower().rstrip(".")
            if contract_text is None or tail in ("", "none", "none yet"):
                out.append(("REV_CONTRACT", p.key))
    known = {p.key for p in before}
    out += [("REV_UNKNOWN", str(k)) for k in entries if k not in known]
    return out
