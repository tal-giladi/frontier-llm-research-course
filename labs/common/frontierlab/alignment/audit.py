"""The student audit: pre-stated thresholds, the rule-out decision, and a report that states its limits
(lesson 18.4 and the Module 18 project).

What the published frameworks share, at the level the course uses (lesson 18.4 has the sources):

* a **capability threshold** is a level of capability at which stronger safeguards are required;
* a model is treated as below a threshold only when evaluations **rule it out**; when they cannot, the developer
  either applies the safeguards anyway (Anthropic activated ASL-3 protections for Claude Opus 4 as a precaution) or
  says the result is in a zone of ambiguity (RSP v3); under-elicitation and sandbagging are reasons an evaluation
  can under-state capability (OpenAI's Preparedness Framework lists sandbagging as a research category and asks for
  a conservative upper bound);
* **safeguards** are claimed only with evidence that they work.

:func:`decide` is that logic for one measured quantity with an interval. :func:`validate` refuses a report that
lacks the disclaimer, labels, evidence or limits, that states thresholds after the results, or that claims
deployment readiness. This is a student exercise: nothing it produces is evidence that any model meets or fails
any lab's deployment thresholds, and the report says so.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

DISCLAIMER = ("This is a student audit exercise from a course. It is not evidence that any model meets, or fails, "
              "any developer's deployment thresholds or safety framework, and it must not be cited as such. The "
              "thresholds below were chosen by the student for practice; they are not any lab's thresholds.")
LABELS = ("MEASURED", "PUBLICLY DOCUMENTED", "REASONABLE INDUSTRY PRACTICE", "INFERENCE/SPECULATION", "company claim")
FORBIDDEN = (r"\bsafe to deploy\b", r"\bmeets? (the )?(asl|deployment|preparedness|critical|high)\b",
             r"\bcertif(y|ied|ies)\b", r"\bcompliant with (the )?(rsp|preparedness|frontier safety)\b")


@dataclass
class Threshold:
    name: str
    metric: str                 # key into the evidence dict
    value: float
    kind: str                   # "capability" (concern if at or above) | "safeguard" (required at or above)
    stated_on: str              # ISO date, must not be after the results
    rationale: str = ""


@dataclass
class Evidence:
    metric: str
    estimate: float
    lo: float
    hi: float
    n: int
    method: str                 # how it was measured, including elicitation
    elicitation_ok: bool = True
    label: str = "MEASURED"
    source: str = ""            # a file path or URL


def decide(ev: Evidence, th: Threshold) -> str:
    """For a capability threshold: "reached" if the interval's lower bound is at or above the threshold,
    "ruled out" if its upper bound is below it *and* elicitation was adequate, else "cannot rule out".
    For a safeguard threshold: "met" if the lower bound is at or above it, "not met" if the upper bound is below,
    else "inconclusive"."""
    if th.kind == "capability":
        if ev.lo >= th.value:
            return "reached"
        if ev.hi < th.value and ev.elicitation_ok:
            return "ruled out"
        return "cannot rule out"
    if th.kind == "safeguard":
        if ev.lo >= th.value:
            return "met"
        if ev.hi < th.value:
            return "not met"
        return "inconclusive"
    raise ValueError(th.kind)


@dataclass
class Report:
    model: str
    date: str
    thresholds: list[Threshold]
    evidence: list[Evidence]
    claims: list[dict] = field(default_factory=list)       # {"text", "label", "source"}
    limits: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    def decisions(self) -> list[dict]:
        ev = {e.metric: e for e in self.evidence}
        return [{"threshold": t.name, "metric": t.metric, "value": t.value, "kind": t.kind,
                 "decision": decide(ev[t.metric], t) if t.metric in ev else "no evidence"} for t in self.thresholds]


def validate(r: Report) -> list[str]:
    """Problems that make the report unacceptable (empty list = acceptable)."""
    problems = []
    if DISCLAIMER not in r.disclaimer:
        problems.append("the disclaimer is missing or edited")
    if len(r.limits) < 3:
        problems.append("fewer than 3 stated limits")
    metrics = {e.metric for e in r.evidence}
    for t in r.thresholds:
        if t.stated_on > r.date:
            problems.append(f"threshold {t.name!r} was stated after the results ({t.stated_on} > {r.date})")
        if t.metric not in metrics:
            problems.append(f"threshold {t.name!r} has no evidence")
    for e in r.evidence:
        if e.label not in LABELS:
            problems.append(f"evidence {e.metric!r} has no valid label")
        if not e.source:
            problems.append(f"evidence {e.metric!r} has no source")
        if not (e.lo <= e.estimate <= e.hi):
            problems.append(f"evidence {e.metric!r}: estimate outside its interval")
    for c in r.claims:
        if c.get("label") not in LABELS:
            problems.append(f"claim without a valid label: {c.get('text', '')[:60]!r}")
        if not c.get("source"):
            problems.append(f"claim without a source: {c.get('text', '')[:60]!r}")
    text = " ".join([c.get("text", "") for c in r.claims] + r.limits).lower()
    for pat in FORBIDDEN:
        if re.search(pat, text):
            problems.append(f"the report claims deployment readiness or framework compliance ({pat})")
    return problems


def render(r: Report) -> str:
    """The audit report as markdown."""
    lines = [f"# Student audit: {r.model}", "", f"> [!IMPORTANT]\n> {r.disclaimer}", "", f"Date of results: {r.date}.", "",
             "## Thresholds (stated before the results)", "", "| Threshold | Kind | Metric | Value | Stated on | Rationale |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {t.name} | {t.kind} | `{t.metric}` | {t.value:g} | {t.stated_on} | {t.rationale} |" for t in r.thresholds]
    lines += ["", "## Evidence", "", "| Metric | Estimate [95% interval] | n | Method and elicitation | Label | Source |",
              "|---|---|---|---|---|---|"]
    lines += [f"| `{e.metric}` | {e.estimate:.3f} [{e.lo:.3f}, {e.hi:.3f}] | {e.n} | {e.method}"
              f"{'' if e.elicitation_ok else ' (elicitation judged inadequate)'} | {e.label} | {e.source} |" for e in r.evidence]
    lines += ["", "## Decisions", "", "| Threshold | Decision |", "|---|---|"]
    lines += [f"| {d['threshold']} | {d['decision']} |" for d in r.decisions()]
    if r.claims:
        lines += ["", "## Other claims", ""] + [f"- {c['text']} ({c['label']}; {c['source']})" for c in r.claims]
    lines += ["", "## Limits", ""] + [f"- {x}" for x in r.limits]
    return "\n".join(lines) + "\n"


def as_dict(r: Report) -> dict:
    return {**asdict(r), "decisions": r.decisions(), "problems": validate(r)}


# --------------------------------------------------------------------------- the three frameworks, as read

FRAMEWORKS = {
    "Anthropic Responsible Scaling Policy": {
        "version": "3.4", "effective": "2026-07-08", "url": "https://www.anthropic.com/rsp-updates",
        "levels": "AI Safety Level standards (ASL-2 baseline; ASL-3 security and deployment standards); capability "
                  "thresholds for chemical and biological weapons, cyber and automated AI R&D",
        "reports": "Risk Reports every 3 to 6 months with external review (from v3.0, 2026-02-24); system cards",
        "decides": "Anthropic (Responsible Scaling Officer); the Long-Term Benefit Trust may request external review "
                   "(v3.2)",
        "note": "v3.0 was a rewrite that says pre-set thresholds proved more ambiguous than expected and adds a zone of "
                "ambiguity and a nonbinding Frontier Safety Roadmap",
    },
    "OpenAI Preparedness Framework": {
        "version": "2", "effective": "2025-04-15",
        "url": "https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf",
        "levels": "Tracked Categories (biological and chemical, cybersecurity, AI self-improvement) with High and "
                  "Critical thresholds; Research Categories (long-range autonomy, sandbagging, autonomous replication "
                  "and adaptation, undermining safeguards, nuclear and radiological)",
        "reports": "Capabilities Reports and Safeguards Reports; system cards at deploymentsafety.openai.com",
        "decides": "Safety Advisory Group recommends, OpenAI leadership decides, the board's Safety and Security "
                   "Committee oversees",
        "note": "High requires safeguards before deployment; Critical also requires safeguards during development",
    },
    "Google DeepMind Frontier Safety Framework": {
        "version": "3.1", "effective": "2026-04-17",
        "url": "https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/strengthening-our-frontier-safety-"
               "framework/frontier-safety-framework_3-1.pdf",
        "levels": "Critical Capability Levels (severe risk) with alert thresholds, and Tracked Capability Levels "
                  "(significant risk) with early-warning evaluations; misuse domains CBRN, cyber, harmful "
                  "manipulation; ML R&D and misalignment (instrumental reasoning) in one section",
        "reports": "Frontier Safety Framework reports per model (for example the Gemini 3.7 Flash report, August 2026)",
        "decides": "safety case reviews before external launch and large internal deployments; a governance section "
                   "(v3.1)",
        "note": "v3.0 (2025-09-22) added the harmful-manipulation CCL",
    },
}


__all__ = ["DISCLAIMER", "LABELS", "Threshold", "Evidence", "decide", "Report", "validate", "render", "as_dict",
           "FRAMEWORKS"]
