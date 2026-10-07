"""The capstone package and its checker (lessons 20.1 and 20.2, the Module 20 project).

A capstone package is one folder:

    claim.yaml        the claim id, the arms and their roles, seeds, comparisons and what each changes,
                      the comparison axis, metric, margin, tuning budget per arm, external parents
    contract.md       the filled experiment contract (templates/experiment-contract.md headings)
    runs/<arm>-s<seed>/run_card.yaml    one run card per arm and seed (written by the course loop)
    results.json      per comparison: per-seed values, mean difference, interval, method, decision;
                      the baseline's noise floor
    claims.yaml       every claim the report makes: text, comparison, direction, scope, evidence label
    report.md         the write-up (must have a Limits section)

:func:`check_package` returns a list of :class:`Problem`. A package is sound when the list is empty. The
checks are the rubric's soundness criteria made mechanical; none of them asks whether the new method won:

* **contract** — every section present and filled, a hypothesis status, a named axis that matches
  ``claim.yaml``, a decision rule with a number, every correctness check ticked;
* **run cards** — one per arm and seed, each with a parent that is another run of the package or a declared
  external parent (Module 1's experiment record: the chain must be traceable);
* **parity** — for every comparison and seed, :func:`frontierlab.record.diff_cards` finds nothing that
  invalidates it on the declared axis, given the declared changed variables; equal tuning budgets; the same seeds;
* **uncertainty** — at least two seeds per compared arm, an interval that contains its mean and names its method,
  a mean that matches the per-seed values, a decision that follows from the interval and the margin, and the
  baseline's noise floor;
* **claims within evidence** — every claim has an evidence label; a MEASURED claim names a comparison, says no
  more than the recomputed decision allows and is scoped to the course scale; a PUBLICLY DOCUMENTED claim names
  its source; the report has a Limits section and no proof language.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from frontierlab.capstone.uncertainty import DECISIONS, decide
from frontierlab.record.diff import diff_cards

CONTRACT_SECTIONS = ("Question and decision", "Hypothesis", "Baseline", "What changes and what is held fixed",
                     "Comparison axis", "Budget", "Metrics and decision rule", "Correctness checks",
                     "Fallback evidence", "Limits of the conclusion", "Changes after results")
STATUSES = ("established effect", "reported effect", "may not appear at this scale")
AXIS_WORDS = {"tokens": "equal tokens", "params": "equal parameters", "flops": "equal training flops",
              "wallclock": "equal wall-clock"}
ROLES = ("baseline", "reproduction", "extension", "control")
LABELS = ("MEASURED", "PUBLICLY DOCUMENTED", "INFERENCE/SPECULATION", "PROJECTED", "ANALYSIS")
DIRECTIONS = {"equivalent": ("equivalent", "no-difference-detected"),
              "a_lower": ("lower",),
              "a_higher": ("higher",),
              "inconclusive": ("no-difference-detected", "inconclusive")}
OVERCLAIM = (r"\bprov(e|es|ed|en|ing)\b", r"\bproof\b", r"\bdefinitive(ly)?\b", r"\bguarantee[sd]?\b",
             r"\bestablishes that\b", r"\bconclusively\b")
PLACEHOLDER = re.compile(r"TODO|<fill|\(Example:|path or name\.|one sentence\.$")
EMPTY_FIELD = re.compile(r"^\s*-\s*\*\*[^*]+:\*\*\s*$")


@dataclass(frozen=True)
class Problem:
    code: str
    where: str
    message: str

    @property
    def key(self) -> str:
        return f"{self.code}@{self.where}"

    def __str__(self) -> str:
        return f"{self.code:18s} {self.where}: {self.message}"


# --------------------------------------------------------------------------- loading

def _yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None


def load(pkg) -> dict:
    """All parts of a package as plain data (missing parts are None)."""
    pkg = Path(pkg)
    res = pkg / "results.json"
    return {"dir": pkg, "claim": _yaml(pkg / "claim.yaml"), "claims": _yaml(pkg / "claims.yaml"),
            "results": json.loads(res.read_text(encoding="utf-8")) if res.exists() else None,
            "contract": (pkg / "contract.md").read_text(encoding="utf-8") if (pkg / "contract.md").exists() else None,
            "report": (pkg / "report.md").read_text(encoding="utf-8") if (pkg / "report.md").exists() else None}


def arm_seeds(claim: dict, arm: str) -> list[int]:
    return list(claim["arms"][arm].get("seeds", claim.get("seeds", [])))


def run_name(arm: str, seed: int) -> str:
    return f"{arm}-s{seed}"


# --------------------------------------------------------------------------- contract

def contract_sections(text: str) -> dict[str, str]:
    """Body of each ``## heading`` of a contract, keyed by the template's section name it starts with."""
    out, cur, buf = {}, None, []
    for line in text.splitlines():
        if line.startswith("## "):
            if cur is not None:
                out[cur] = "\n".join(buf).strip()
            head = line[3:].strip()
            cur = next((s for s in CONTRACT_SECTIONS if head.lower().startswith(s.lower())), head)
            buf = []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf).strip()
    return out


def contract_problems(text: str | None, axis: str | None = None) -> list[Problem]:
    if text is None:
        return [Problem("CONTRACT_MISSING", "contract.md", "no experiment contract in the package")]
    secs, out = contract_sections(text), []
    for s in CONTRACT_SECTIONS:
        if s not in secs:
            out.append(Problem("CONTRACT_SECTION", f"contract.md#{s}", "section missing"))
            continue
        body = secs[s]
        if not body or any(EMPTY_FIELD.match(l) or PLACEHOLDER.search(l) for l in body.splitlines()):
            out.append(Problem("CONTRACT_EMPTY", f"contract.md#{s}", "section empty or still holds template text"))
    hyp = secs.get("Hypothesis", "").lower()
    if "Hypothesis" in secs and not any(st in hyp for st in STATUSES):
        out.append(Problem("CONTRACT_STATUS", "contract.md#Hypothesis",
                           f"hypothesis status must be one of: {', '.join(STATUSES)}"))
    ax = secs.get("Comparison axis", "").lower()
    named = [k for k, w in AXIS_WORDS.items() if w in ax]
    if "Comparison axis" in secs and (len(named) == 0 or (axis is not None and axis not in named)):
        out.append(Problem("CONTRACT_AXIS", "contract.md#Comparison axis",
                           f"the axis must be named ({', '.join(AXIS_WORDS.values())}) and match claim.yaml ({axis})"))
    rule = secs.get("Metrics and decision rule", "")
    if "Metrics and decision rule" in secs and not re.search(r"decision rule[^\n]*\d", rule, re.I):
        out.append(Problem("CONTRACT_RULE", "contract.md#Metrics and decision rule",
                           "the decision rule must be stated with its numbers before the runs"))
    for line in secs.get("Correctness checks", "").splitlines():
        if re.match(r"^\s*-\s*\[\s\]", line):
            out.append(Problem("CONTRACT_UNCHECKED", "contract.md#Correctness checks",
                               f"unticked check, results do not count yet: {line.strip()[6:]}"))
    return out


# --------------------------------------------------------------------------- run cards and parity

def runcard_problems(pkg: Path, claim: dict) -> list[Problem]:
    out = []
    names = {run_name(a, s) for a in claim["arms"] for s in arm_seeds(claim, a)}
    external = set(claim.get("external_parents") or [])
    for a in claim["arms"]:
        for s in arm_seeds(claim, a):
            n = run_name(a, s)
            p = pkg / "runs" / n / "run_card.yaml"
            if not p.exists():
                out.append(Problem("RUNCARD_MISSING", f"runs/{n}", "no run_card.yaml"))
                continue
            parent = (_yaml(p) or {}).get("parent_run")
            if not parent or (parent not in names and parent not in external):
                out.append(Problem("RUNCARD_PARENT", f"runs/{n}",
                                   f"parent_run {parent!r} is neither a run of this package nor a declared external parent"))
    return out


def parity_problems(pkg: Path, claim: dict) -> list[Problem]:
    out = []
    axis = claim.get("axis", "tokens")
    tuning = claim.get("tuning") or {}
    for c in claim.get("comparisons", []):
        a, b, where = c["a"], c["b"], f"comparison {c['name']}"
        if a not in claim["arms"] or b not in claim["arms"]:
            out.append(Problem("PKG_ARMS", where, "comparison names an arm that claim.yaml does not define"))
            continue
        if tuning.get(a) != tuning.get(b):
            out.append(Problem("TUNING", where, f"tuning budget differs: {a} {tuning.get(a)} vs {b} {tuning.get(b)} trials"))
        sa, sb = arm_seeds(claim, a), arm_seeds(claim, b)
        if sorted(sa) != sorted(sb):
            out.append(Problem("SEEDS", where, f"different seed sets: {a} {sa} vs {b} {sb}"))
        for s in sorted(set(sa) & set(sb)):
            pa, pb = pkg / "runs" / run_name(a, s), pkg / "runs" / run_name(b, s)
            if not (pa / "run_card.yaml").exists() or not (pb / "run_card.yaml").exists():
                continue
            bad = [f for f in diff_cards(pb, pa, changed=c.get("changed", []), axis=axis) if f.severity == "invalidates"]
            if bad:                                   # one problem per compared pair, listing every finding
                out.append(Problem("PARITY", f"{where} seed {s}",
                                   "; ".join(f"{f.key}: {f.a!r} -> {f.b!r} ({f.reason})" for f in bad)))
    return out


# --------------------------------------------------------------------------- uncertainty

def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not math.isnan(x) and not math.isinf(x)


def recomputed_decisions(claim: dict, results: dict) -> dict[str, str]:
    """Decision per comparison from its reported interval and the margin in claim.yaml (not the reported one)."""
    out = {}
    for r in (results or {}).get("comparisons", []):
        ci = r.get("ci") or [float("nan"), float("nan")]
        try:
            out[r["name"]] = decide(float(r["mean_diff"]), float(ci[0]), float(ci[1]), float(claim.get("margin", 0)))
        except (TypeError, ValueError, KeyError, IndexError):
            out[r.get("name", "?")] = "inconclusive"
    return out


def uncertainty_problems(claim: dict, results: dict | None) -> list[Problem]:
    if results is None:
        return [Problem("RESULTS_MISSING", "results.json", "no results file")]
    out = []
    by_name = {r.get("name"): r for r in results.get("comparisons", [])}
    margin = float(claim.get("margin", 0))
    for c in claim.get("comparisons", []):
        where = f"results.json#{c['name']}"
        r = by_name.get(c["name"])
        if r is None:
            out.append(Problem("UNC_INTERVAL", where, "comparison has no result"))
            continue
        per = r.get("per_seed") or {}
        va, vb = per.get("a") or [], per.get("b") or []
        if min(len(va), len(vb)) < 2:
            out.append(Problem("UNC_SEEDS", where, f"needs at least 2 seeds per arm, has {len(va)} and {len(vb)}"))
        ci = r.get("ci")
        m = r.get("mean_diff")
        if not (isinstance(ci, (list, tuple)) and len(ci) == 2 and all(_finite(x) for x in ci) and _finite(m)
                and ci[0] <= m <= ci[1] and r.get("method")):
            out.append(Problem("UNC_INTERVAL", where, "needs a finite interval that contains the mean and names its method"))
            continue
        if va and vb and len(va) == len(vb) and abs((sum(va) / len(va) - sum(vb) / len(vb)) - m) > 1e-6:
            out.append(Problem("UNC_MEAN", where, "mean_diff does not match the per-seed values"))
        want = decide(m, ci[0], ci[1], margin)
        if r.get("decision") not in DECISIONS or r.get("decision") != want:
            out.append(Problem("UNC_DECISION", where, f"reported {r.get('decision')!r}; the stated rule gives {want!r}"))
    nf = results.get("noise_floor") or {}
    if not (_finite(nf.get("seed_std")) and _finite(nf.get("mde"))):
        out.append(Problem("UNC_NOISE", "results.json#noise_floor", "report the baseline's seed std and the minimum detectable effect"))
    return out


# --------------------------------------------------------------------------- claims

def claim_problems(claims: list | None, decisions: dict[str, str], report: str | None = None) -> list[Problem]:
    """Claims within evidence. ``decisions`` are the *recomputed* decisions per comparison."""
    out = []
    for i, c in enumerate(claims or []):
        where = f"claims.yaml#{c.get('id', i)}"
        label = c.get("label")
        if label not in LABELS:
            out.append(Problem("CLAIM_LABEL", where, f"evidence label must be one of {LABELS}"))
            continue
        if label == "PUBLICLY DOCUMENTED" and not c.get("source"):
            out.append(Problem("CLAIM_SOURCE", where, "a documented claim names its source (paper, section)"))
        if label == "MEASURED":
            comp = c.get("comparison")
            if comp not in decisions:
                out.append(Problem("CLAIM_COMPARISON", where, f"names no comparison of the package ({comp!r})"))
            elif c.get("direction") not in DIRECTIONS[decisions[comp]]:
                out.append(Problem("CLAIM_DIRECTION", where,
                                   f"says {c.get('direction')!r}; the evidence ({decisions[comp]}) allows {DIRECTIONS[decisions[comp]]}"))
            if c.get("scope", "course") != "course":
                out.append(Problem("CLAIM_SCOPE", where, "a course-scale measurement cannot support a claim about another scale"))
    if report is None:
        out.append(Problem("REPORT_MISSING", "report.md", "no report"))
        return out
    for pat in OVERCLAIM:
        for m in re.finditer(pat, report, re.I):
            out.append(Problem("REPORT_OVERCLAIM", "report.md", f"proof language: {m.group(0)!r}"))
    if not re.search(r"^#+\s*Limits", report, re.M | re.I):
        out.append(Problem("REPORT_LIMITS", "report.md", "no Limits section"))
    return out


# --------------------------------------------------------------------------- everything

def role_problems(claim: dict) -> list[Problem]:
    roles = [v.get("role") for v in claim.get("arms", {}).values()]
    out = [Problem("PKG_ROLES", f"claim.yaml#arms.{k}", f"unknown role {v.get('role')!r}")
           for k, v in claim.get("arms", {}).items() if v.get("role") not in ROLES]
    for need, n in (("baseline", 1), ("reproduction", 1), ("extension", 1)):
        if roles.count(need) < n or (need == "baseline" and roles.count(need) != 1):
            out.append(Problem("PKG_ROLES", "claim.yaml#arms", f"needs {'exactly one' if need == 'baseline' else 'at least one'} {need} arm"))
    return out


def check_package(pkg) -> list[Problem]:
    """Every soundness problem of a capstone package (empty list = sound)."""
    P = load(pkg)
    claim = P["claim"]
    if claim is None:
        return [Problem("PKG_CLAIM", "claim.yaml", "no claim.yaml")]
    out = role_problems(claim)
    out += contract_problems(P["contract"], claim.get("axis"))
    out += runcard_problems(P["dir"], claim)
    out += parity_problems(P["dir"], claim)
    out += uncertainty_problems(claim, P["results"])
    out += claim_problems(P["claims"], recomputed_decisions(claim, P["results"]), P["report"])
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Check a capstone package; exit 1 if it has problems.")
    ap.add_argument("package")
    a = ap.parse_args(argv)
    probs = check_package(a.package)
    for p in probs:
        print(p)
    print("SOUND: no problems" if not probs else f"{len(probs)} problem(s)")
    return 1 if probs else 0


__all__ = ["CONTRACT_SECTIONS", "DIRECTIONS", "LABELS", "Problem", "ROLES", "STATUSES", "arm_seeds", "check_package",
           "claim_problems", "contract_problems", "contract_sections", "load", "parity_problems",
           "recomputed_decisions", "role_problems", "run_name", "runcard_problems", "uncertainty_problems"]
