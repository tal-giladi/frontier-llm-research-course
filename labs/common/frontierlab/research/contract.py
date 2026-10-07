"""The experiment contract as data, its validator, and the rubric total (Module 19 project).

The fields follow ``templates/experiment-contract.md`` (plan section 9). A contract is a dict (YAML on disk);
:func:`validate` returns the problems a reviewer would raise before any GPU time is spent:

* every section filled; the hypothesis status one of the three the course uses;
* a comparison axis from the four, and a sentence on what it does not answer;
* a budget labelled PROJECTED or measured, with extra compute (teacher, judge, verifier, generator) stated, and the
  same tuning budget for every arm;
* a noise floor and seed count from which the minimum detectable effect is recomputed
  (``stats.min_detectable_effect``); a contract whose expected effect is below it is **underpowered**;
* a decision rule with a number in it, written before the results;
* fallback evidence when the effect "may not appear at this scale";
* at least three limits, one of them about scale.

:data:`RUBRIC` is ``templates/experiment-rubric.md`` as data; :func:`rubric_total` applies its pass rule
(at least 10 of 14, no criterion at 0).
"""

from __future__ import annotations

import re

from frontierlab.research.proposals import CAPSTONE_CLAIMS
from frontierlab.stats import min_detectable_effect

STATUS = ("established effect", "reported effect", "may not appear at this scale")
AXES = ("tokens", "params", "flops", "wallclock")
REQUIRED = ("claim_id", "question", "decision", "hypothesis", "status", "baseline", "baseline_tuning", "changed",
            "held_fixed", "axis", "axis_does_not_answer", "budget", "metrics", "decision_rule",
            "correctness_checks", "fallback", "limits", "stated_on")
BUDGET_REQUIRED = ("hardware", "gpu_hours", "label", "extra_compute", "tuning_per_arm")
METRICS_REQUIRED = ("primary", "uncertainty", "noise_floor", "n_seeds", "expected_effect")

RUBRIC = (
    ("question", "Question and decision"),
    ("controls", "Controls"),
    ("axis_budget", "Comparison axis and budget parity"),
    ("correctness", "Correctness"),
    ("uncertainty", "Uncertainty"),
    ("conclusion", "Conclusion matches evidence"),
    ("limits", "Limits"),
)


def _empty(v) -> bool:
    return v is None or (isinstance(v, (str, list, dict, tuple)) and len(v) == 0) or (isinstance(v, str) and not v.strip())


def mde(contract: dict) -> float:
    """Minimum detectable effect implied by the contract's noise floor and seeds per arm (80% power, alpha 0.05)."""
    m = contract["metrics"]
    return min_detectable_effect(float(m["noise_floor"]), int(m["n_seeds"]))


def validate(c: dict, results_on: str | None = None, capstone: bool = True) -> list[str]:
    """Problems with a contract (empty list: ready to run). ``results_on``: ISO date of the first result, if any."""
    probs = []
    for k in REQUIRED:
        if _empty(c.get(k)):
            probs.append(f"{k}: missing or empty")
    if probs:
        return probs
    if capstone and c["claim_id"] not in CAPSTONE_CLAIMS:
        probs.append(f"claim_id: '{c['claim_id']}' is not on the capstone list {sorted(CAPSTONE_CLAIMS)}")
    if not str(c["question"]).strip().endswith("?"):
        probs.append("question: one answerable question, ending in '?'")
    if c["status"] not in STATUS:
        probs.append(f"status: one of {STATUS}")
    if c["axis"] not in AXES:
        probs.append(f"axis: one of {AXES}")
    b = c["budget"]
    for k in BUDGET_REQUIRED:
        if _empty(b.get(k)) and k != "extra_compute":
            probs.append(f"budget.{k}: missing")
    if "extra_compute" not in b:
        probs.append("budget.extra_compute: state teacher/judge/verifier/generator compute, or 'none'")
    if str(b.get("label", "")) not in ("PROJECTED", "measured"):
        probs.append("budget.label: PROJECTED (with its formula) or measured")
    elif b.get("label") == "PROJECTED" and _empty(b.get("formula")):
        probs.append("budget.formula: a PROJECTED budget needs the formula it comes from")
    tune = b.get("tuning_per_arm")
    if isinstance(tune, dict) and len(set(map(str, tune.values()))) > 1:
        probs.append(f"budget.tuning_per_arm: arms get different tuning budgets {tune}; the baseline gets the same")
    m = c["metrics"]
    for k in METRICS_REQUIRED:
        if _empty(m.get(k)):
            probs.append(f"metrics.{k}: missing")
    if not any(p.startswith("metrics.") for p in probs):
        need = mde(c)
        if int(m["n_seeds"]) < 2:
            probs.append("metrics.n_seeds: at least 2 per arm")
        if abs(float(m["expected_effect"])) < need:
            probs.append(f"metrics: underpowered: the expected effect {float(m['expected_effect']):g} is below the "
                         f"minimum detectable effect {need:.3g} for noise {float(m['noise_floor']):g} and "
                         f"{int(m['n_seeds'])} seeds per arm; add seeds, raise the effect, or state it as a pilot")
    if not re.search(r"\d", str(c["decision_rule"])):
        probs.append("decision_rule: needs a threshold you can check (a number)")
    if c["status"] == "may not appear at this scale" and str(c["fallback"]).strip().lower() in ("none", "n/a", "-"):
        probs.append("fallback: an effect that may not appear needs fallback evidence (provided traces, published curves)")
    limits = c["limits"] if isinstance(c["limits"], list) else [c["limits"]]
    if len(limits) < 3:
        probs.append("limits: state at least three")
    if not any(re.search(r"scale|size|param", str(x), re.I) for x in limits):
        probs.append("limits: one limit must name the model scale")
    if results_on is not None and str(c["stated_on"]) > str(results_on):
        probs.append("stated_on: the contract is dated after the first result")
    return probs


def validate_proposal(c: dict, results_on: str | None = None) -> list[str]:
    """The capstone proposal of the Module 19 project: a valid contract (:func:`validate`) plus the claim's source,
    what counts as reproduced (lesson 19.2), one extension, the cheap proxy and the kill criterion (lesson 19.1)."""
    probs = validate(c, results_on=results_on, capstone=True)
    if not re.search(r"https?://", str(c.get("source", ""))):
        probs.append("source: the paper, the section or figure of the claim, and its URL")
    r = c.get("reproduction") or {}
    for k in ("direction", "tolerance_rule", "magnitude"):
        if _empty(r.get(k)):
            probs.append(f"reproduction.{k}: missing")
    tol = r.get("tolerance")
    if tol is None or not isinstance(tol, (int, float)) or tol <= 0:
        probs.append("reproduction.tolerance: a positive number, the smallest effect that counts")
    else:
        try:
            if float(tol) < float(c["metrics"]["noise_floor"]):
                probs.append("reproduction.tolerance: below the noise floor; an effect that small cannot be told from noise")
        except (KeyError, TypeError, ValueError):
            pass
    for k in ("extension", "proxy", "kill"):
        if _empty(c.get(k)):
            probs.append(f"{k}: missing")
    return probs


def rubric_total(scores: dict) -> dict:
    """``scores``: {criterion key: 0, 1 or 2} for every key of RUBRIC. Pass at >= 10 of 14 with no 0."""
    keys = [k for k, _ in RUBRIC]
    missing = [k for k in keys if k not in scores]
    if missing:
        raise ValueError(f"score every criterion; missing {missing}")
    if any(scores[k] not in (0, 1, 2) for k in keys):
        raise ValueError("scores are 0, 1 or 2")
    total = sum(scores[k] for k in keys)
    zeros = [k for k in keys if scores[k] == 0]
    return {"total": total, "max": 2 * len(keys), "zeros": zeros, "passed": total >= 10 and not zeros}
