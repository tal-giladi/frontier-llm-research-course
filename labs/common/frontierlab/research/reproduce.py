"""Reproducing a published claim (lesson 19.2).

Three pieces:

1. :func:`lr_sensitivity` — Wortsman et al. (arXiv 2309.14322, section 2.2): over a sweep of learning rates,
   ``E_eta[min(l(eta), l0) - l*]`` with ``l*`` the best loss of the sweep and ``l0`` the loss at initialisation, so a
   diverged run counts as ``l0`` and never as infinity.
2. :func:`decide` — the reproduction decision, made against a tolerance stated *before* the runs. Effects are
   signed so that positive means "in the direction the paper claims". With per-seed effects ``e_s`` and a
   t-interval ``[lo, hi]`` over seeds:

   ==========================  ================================================================================
   direction outcome           rule
   ==========================  ================================================================================
   ``reproduced``              ``lo > 0`` and ``mean >= tolerance``: the claimed direction, at least as large as
                               the smallest effect you said would count
   ``contradicted``            ``hi < 0``: the opposite direction, outside the noise
   ``not reproduced``          ``hi < tolerance``: an effect as large as the tolerance is excluded
   ``inconclusive``            otherwise (the interval straddles the tolerance): more seeds, not a verdict
   ==========================  ================================================================================

   Magnitude is decided separately and only when the published number is on a comparable footing (same metric,
   same units, a scale you matched): ``|mean - published| <= magnitude_tol`` is ``matches``, otherwise
   ``differs``; with no comparable published number it is ``not comparable``. "Direction reproduced, magnitude not
   comparable" is the normal outcome of a small-scale reproduction and must be written that way.
3. :class:`Deviation`, :class:`Record` and :func:`validate` — everything that differs from the paper, why, and
   what you expect it to do to the result, recorded before the results; a record whose tolerance is dated after
   its results, whose outcome does not follow from its numbers, or that omits an aspect is refused.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from frontierlab.stats import summary

# two-sided 95% Student t quantiles, df = 1..30 (beyond: normal)
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
        2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045,
        2.042]

ASPECTS = ("scale", "data", "tokens", "optimizer", "learning rates", "evaluation", "seeds", "code")
EXPECTED = ("weakens", "strengthens", "unknown", "none expected")


def lr_sensitivity(losses: dict, l0: float) -> float:
    """Wortsman et al. section 2.2 for one sweep: ``losses`` maps learning rate -> final loss (``nan`` or ``inf`` for a
    diverged run). Returns ``mean_eta(min(l, l0)) - min_eta(min(l, l0))``."""
    vals = []
    for v in losses.values():
        v = float(v)
        vals.append(l0 if not math.isfinite(v) else min(v, l0))
    if not vals:
        raise ValueError("empty sweep")
    a = np.asarray(vals, dtype=np.float64)
    return float(a.mean() - a.min())


def t_interval(values) -> tuple[float, float, float]:
    """(mean, lo, hi): 95% Student-t interval over seeds (needs at least 2 values)."""
    s = summary(values)
    if s["n"] < 2:
        raise ValueError("a seed interval needs at least 2 seeds")
    t = T975[s["n"] - 2] if s["n"] - 1 <= len(T975) else 1.96
    h = t * s["sem"]
    return s["mean"], s["mean"] - h, s["mean"] + h


def decide(effects, tolerance: float, published: float | None = None, magnitude_tol: float | None = None) -> dict:
    """The reproduction decision (rules in the module docstring). ``effects``: per-seed signed effects."""
    if tolerance < 0:
        raise ValueError("tolerance is the smallest effect that counts; it cannot be negative")
    mean, lo, hi = t_interval(effects)
    if lo > 0 and mean >= tolerance:
        direction = "reproduced"
    elif hi < 0:
        direction = "contradicted"
    elif hi < tolerance:
        direction = "not reproduced"
    else:
        direction = "inconclusive"
    if published is None or magnitude_tol is None:
        magnitude = "not comparable"
    else:
        magnitude = "matches" if abs(mean - published) <= magnitude_tol else "differs"
    return {"direction": direction, "magnitude": magnitude, "mean": mean, "ci": (lo, hi), "n": len(list(effects)),
            "tolerance": tolerance, "published": published, "magnitude_tol": magnitude_tol}


@dataclass
class Deviation:
    aspect: str               # one of ASPECTS
    paper: str                # what the paper did
    ours: str                 # what we did
    reason: str               # why
    expected_effect: str      # one of EXPECTED, about the claim being tested
    recorded: str = "before"  # "before" or "after" the results
    note: str = ""            # required when recorded after the results


@dataclass
class Record:
    claim: str                         # the claim in the paper's words or a faithful paraphrase
    source: str                        # paper + section/figure where the claim is made
    url: str
    tolerance: float
    tolerance_rule: str                # how the tolerance follows from the noise, in words
    stated_on: str                     # ISO date the tolerance and rule were written
    results_on: str                    # ISO date of the first result
    effects: list                      # per-seed signed effects
    outcome: dict                      # decide(...) as reported
    deviations: list = field(default_factory=list)
    author_contact: str = ""           # "not needed: ..." or "asked on <date>: <question>; answer: ..."
    scope: str = ""                    # one sentence: at what scale/data/setting the conclusion holds

    def to_dict(self) -> dict:
        return asdict(self)


def validate(rec: Record, aspects=ASPECTS) -> list[str]:
    """Problems with a reproduction record (empty list: acceptable)."""
    probs = []
    if not rec.claim.strip() or not rec.source.strip() or not rec.url.startswith("http"):
        probs.append("claim: give the claim, its location in the paper (section or figure) and the URL")
    if rec.stated_on > rec.results_on:
        probs.append("tolerance: stated after the first result; a tolerance chosen after seeing results decides nothing")
    if not rec.tolerance_rule.strip():
        probs.append("tolerance: say how it follows from the seed noise")
    try:
        again = decide(rec.effects, rec.tolerance, rec.outcome.get("published"), rec.outcome.get("magnitude_tol"))
        if again["direction"] != rec.outcome.get("direction"):
            probs.append(f"outcome: the numbers give '{again['direction']}', the record says '{rec.outcome.get('direction')}'")
        if again["magnitude"] != rec.outcome.get("magnitude"):
            probs.append(f"outcome: magnitude is '{again['magnitude']}', the record says '{rec.outcome.get('magnitude')}'")
    except ValueError as e:
        probs.append(f"outcome: {e}")
    covered = {d.aspect for d in rec.deviations}
    for a in aspects:
        if a not in covered:
            probs.append(f"deviations: no entry for '{a}' (write 'same as the paper' if nothing differs)")
    for d in rec.deviations:
        if d.aspect not in ASPECTS:
            probs.append(f"deviations: unknown aspect '{d.aspect}'")
        if not d.reason.strip():
            probs.append(f"deviations: '{d.aspect}' has no reason")
        if d.expected_effect not in EXPECTED:
            probs.append(f"deviations: '{d.aspect}' expected effect must be one of {EXPECTED}")
        if d.recorded == "after" and not d.note.strip():
            probs.append(f"deviations: '{d.aspect}' was recorded after the results; say what prompted it")
    if not rec.author_contact.strip():
        probs.append("author contact: say whether you contacted the authors, or why it was not needed")
    if not rec.scope.strip():
        probs.append("scope: one sentence on the scale and setting the conclusion covers")
    elif rec.outcome.get("direction") == "reproduced" and any(d.aspect == "scale" and d.paper != d.ours
                                                              for d in rec.deviations):
        if not any(w in rec.scope.lower() for w in ("at this scale", "at our scale", "parameters", "toy")):
            probs.append("scope: a reproduction at a different scale must name the scale it holds at")
    return probs


def render(rec: Record) -> str:
    """The reproduction record as markdown."""
    o = rec.outcome
    lines = [f"# Reproduction record", "", f"**Claim.** {rec.claim}", f"**Source.** {rec.source} — {rec.url}", "",
             "## What counts as reproduced (stated before the runs)", "",
             f"Tolerance {rec.tolerance:.4g}: {rec.tolerance_rule} Stated {rec.stated_on}; first result {rec.results_on}.",
             "", "## Result", "",
             f"Per-seed effects (positive = the paper's direction): {', '.join(f'{e:+.4f}' for e in rec.effects)}.",
             f"Mean {o['mean']:+.4f}, 95% t-interval [{o['ci'][0]:+.4f}, {o['ci'][1]:+.4f}] over {o['n']} seeds.",
             f"**Direction: {o['direction']}. Magnitude: {o['magnitude']}.**", "", f"Scope: {rec.scope}", "",
             "## Deviation log", "", "| Aspect | Paper | Ours | Why | Expected effect | Recorded |", "|---|---|---|---|---|---|"]
    for d in rec.deviations:
        lines.append(f"| {d.aspect} | {d.paper} | {d.ours} | {d.reason} | {d.expected_effect} | {d.recorded}"
                     + (f" ({d.note})" if d.note else "") + " |")
    lines += ["", f"**Author contact.** {rec.author_contact}", ""]
    return "\n".join(lines)
