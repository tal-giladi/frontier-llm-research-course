"""Review and defence of a capstone package (lesson 20.2).

* :data:`REVISION_KIND` — what each kind of problem needs: ``rerun`` (new runs: the evidence itself is not
  comparable or too thin), ``reanalyse`` (the runs are fine, the numbers drawn from them are not), ``rewrite``
  (the evidence is fine, the words claim more than it supports) or ``declare`` (the record hides something that
  must be stated, such as an unticked check or a change after results).
* :func:`questions` — reviewer questions for a package: the generic ones every reviewer asks (rubric), the ones
  specific to the claim (:mod:`frontierlab.capstone.claims`), and one per problem the checker finds, worded as a
  reviewer would ask it. Each has a stable id (``G1``, ``C2``, ``P3``) so the defence and the revision log can
  refer to it.
* :func:`defence_problems` — a written defence answers every question under its own ``### <id>`` heading, with
  evidence: a file of the package or a number with its interval.
* :func:`revision_log_problems` — every problem found before revision is either fixed, with a log entry of the
  right kind, or declined with a reason (blocking problems cannot be declined); rerun entries name the new runs;
  an entry that changed a result says the contract's "Changes after results" was updated.
"""

from __future__ import annotations

import re
from pathlib import Path

from frontierlab.capstone import claims as CL
from frontierlab.capstone.package import Problem, check_package, load

REVISION_KIND = {
    "PKG_CLAIM": "declare", "PKG_ROLES": "declare", "PKG_ARMS": "declare",
    "CONTRACT_MISSING": "declare", "CONTRACT_SECTION": "declare", "CONTRACT_EMPTY": "declare",
    "CONTRACT_STATUS": "declare", "CONTRACT_AXIS": "declare", "CONTRACT_RULE": "declare",
    "CONTRACT_UNCHECKED": "declare",
    "RUNCARD_MISSING": "rerun", "RUNCARD_PARENT": "declare",
    "PARITY": "rerun", "TUNING": "rerun", "SEEDS": "rerun", "UNC_SEEDS": "rerun",
    "RESULTS_MISSING": "reanalyse", "UNC_INTERVAL": "reanalyse", "UNC_MEAN": "reanalyse",
    "UNC_DECISION": "reanalyse", "UNC_NOISE": "reanalyse",
    "CLAIM_LABEL": "rewrite", "CLAIM_SOURCE": "rewrite", "CLAIM_COMPARISON": "rewrite",
    "CLAIM_DIRECTION": "rewrite", "CLAIM_SCOPE": "rewrite",
    "REPORT_MISSING": "rewrite", "REPORT_OVERCLAIM": "rewrite", "REPORT_LIMITS": "rewrite",
}
KINDS = ("rerun", "reanalyse", "rewrite", "declare")
# Problems that make the result not count until fixed: they cannot be declined in a revision.
BLOCKING = ("PARITY", "TUNING", "SEEDS", "UNC_SEEDS", "UNC_INTERVAL", "UNC_DECISION", "CLAIM_DIRECTION",
            "CLAIM_SCOPE", "CONTRACT_UNCHECKED", "RUNCARD_MISSING")

ASK = {
    "CONTRACT": "Your contract is incomplete at {where} ({message}). Was this stated before the runs, and if not, what changed?",
    "RUNCARD_MISSING": "{where} has no run card. How can a reader reproduce or compare that run?",
    "RUNCARD_PARENT": "{where} names no traceable parent ({message}). What did this run start from?",
    "PARITY": "In {where}, {message}. Why is this still a comparison on your declared axis, and what would rerunning change?",
    "TUNING": "In {where}, {message}. How do you know the baseline was tuned as hard as the arm you favour?",
    "SEEDS": "In {where}, {message}. How are the arms paired if they do not share seeds?",
    "UNC_SEEDS": "In {where}, {message}. What is the seed noise of that arm, and how would you know?",
    "UNC_INTERVAL": "In {where}, {message}. What is the uncertainty of this number?",
    "UNC_MEAN": "In {where}, {message}. Which number is right, and how did the other one get into the results?",
    "UNC_DECISION": "In {where}, {message}. Which decision does your pre-stated rule give, and does the report follow it?",
    "UNC_NOISE": "What is the baseline's seed-to-seed noise, and what is the smallest effect this design could detect?",
    "CLAIM_DIRECTION": "Claim {where}: {message}. Can you state it so that it says only what the interval supports?",
    "CLAIM_SCOPE": "Claim {where} is about another scale ({message}). What evidence at that scale do you have?",
    "CLAIM_LABEL": "Claim {where} has no evidence label. Is it measured, documented, projected or inference?",
    "CLAIM_SOURCE": "Claim {where} cites no source. Where exactly does the paper say this?",
    "CLAIM_COMPARISON": "Claim {where} is labelled MEASURED but names no comparison. Which runs support it?",
    "REPORT_OVERCLAIM": "The report uses {message}. What would it take to justify that word?",
    "REPORT_LIMITS": "The report has no Limits section. At what scale, data and architecture does the conclusion stop?",
    "REPORT_MISSING": "There is no report. What is the claim?",
    "RESULTS_MISSING": "There is no results file. Where are the numbers?",
}


def revision_kind(code: str) -> str:
    """The kind of revision a problem code needs (KeyError for an unknown code)."""
    return REVISION_KIND[code]


def _ask(p: Problem) -> str:
    key = "CONTRACT" if p.code.startswith("CONTRACT") else p.code
    tmpl = ASK.get(key, "{where}: {message}. How do you answer this?")
    return tmpl.format(where=p.where, message=p.message)


def questions(pkg, problems: list[Problem] | None = None) -> list[dict]:
    """Reviewer questions for a package: generic (G), claim-specific (C), one per problem (P)."""
    P = load(pkg)
    problems = check_package(pkg) if problems is None else problems
    out = [{"id": f"G{i}", "kind": "generic", "question": q, "trigger": "rubric"}
           for i, q in enumerate(CL.GENERIC_QUESTIONS, 1)]
    cid = (P["claim"] or {}).get("claim")
    if cid in CL.CLAIMS:
        out += [{"id": f"C{i}", "kind": "claim", "question": q, "trigger": cid}
                for i, q in enumerate(CL.CLAIMS[cid].questions, 1)]
    out += [{"id": f"P{i}", "kind": "problem", "question": _ask(p), "trigger": p.key, "revision": revision_kind(p.code)}
            for i, p in enumerate(problems, 1)]
    return out


def defence_sections(text: str) -> dict[str, str]:
    """``### <id>`` sections of a defence (the heading may continue after the id; quoted ``>`` lines are dropped)."""
    out, cur, buf = {}, None, []
    for line in text.splitlines():
        m = re.match(r"^###\s+([A-Z]\d+)\b", line)
        if m:
            if cur is not None:
                out[cur] = "\n".join(buf).strip()
            cur, buf = m.group(1), []
        elif line.startswith("#"):
            if cur is not None:
                out[cur] = "\n".join(buf).strip()
            cur, buf = None, []
        elif cur is not None and not line.lstrip().startswith(">"):     # a quoted question is not an answer
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf).strip()
    return out


EVIDENCE = re.compile(r"(\b[\w./-]+\.(json|yaml|md|jsonl)\b|runs/[\w.-]+|\[[+-]?\d*\.?\d+\s*,\s*[+-]?\d*\.?\d+\])")


def defence_problems(text: str, qs: list[dict]) -> list[Problem]:
    """Every question answered under its id, at least two sentences, with evidence (a package file or an interval)."""
    secs, out = defence_sections(text), []
    for q in qs:
        body = secs.get(q["id"])
        if body is None:
            out.append(Problem("DEF_MISSING", q["id"], "no answer"))
            continue
        if len(re.findall(r"[.!?](\s|$)", body)) < 2 or re.fullmatch(r"(?i)\W*(n/?a|none|tbd)\W*", body):
            out.append(Problem("DEF_EMPTY", q["id"], "answer in at least two sentences"))
        if not EVIDENCE.search(body):
            out.append(Problem("DEF_EVIDENCE", q["id"], "cite a package file or a number with its interval"))
    return out


def revision_log_problems(log: list[dict], before: list[Problem], after: list[Problem], pkg=None,
                          contract_text: str | None = None) -> list[Problem]:
    """Check a revision log (a list of entries ``{problem, kind, change, files, result_changed, reason}``)."""
    out = []
    entries = {e.get("problem"): e for e in (log or [])}
    still = {p.key for p in after}
    for p in before:
        e = entries.get(p.key)
        if e is None:
            out.append(Problem("REV_UNLOGGED", p.key, "no revision-log entry for this problem"))
            continue
        kind = e.get("kind")
        if kind == "decline":
            if p.code in BLOCKING:
                out.append(Problem("REV_DECLINE", p.key, "a blocking problem cannot be declined"))
            elif len((e.get("reason") or "").split()) < 8:
                out.append(Problem("REV_DECLINE", p.key, "a declined problem needs a reason (one sentence or more)"))
            continue
        if kind != revision_kind(p.code):
            out.append(Problem("REV_KIND", p.key, f"logged as {kind!r}; this problem needs {revision_kind(p.code)!r}"))
        if p.key in still:
            out.append(Problem("REV_NOT_FIXED", p.key, "logged as fixed but the checker still finds it"))
        if kind == "rerun":
            files = e.get("files") or []
            if not files or (pkg is not None and not all((Path(pkg) / f).exists() for f in files)):
                out.append(Problem("REV_FILES", p.key, "a rerun names the new runs, and they must exist"))
        if e.get("result_changed"):
            tail = (contract_text or "").split("## Changes after results")[-1].strip().lower().rstrip(".")
            if contract_text is None or tail in ("", "none", "none yet"):
                out.append(Problem("REV_CONTRACT", p.key, "a changed result must be recorded under 'Changes after results'"))
    known = {p.key for p in before}
    for k in entries:
        if k not in known:
            out.append(Problem("REV_UNKNOWN", str(k), "entry for a problem the checker did not find before revision"))
    return out


__all__ = ["BLOCKING", "KINDS", "REVISION_KIND", "defence_problems", "defence_sections", "questions",
           "revision_kind", "revision_log_problems"]
