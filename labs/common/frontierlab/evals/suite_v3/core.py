"""Shared pieces of Eval Suite v3: a v2 result plus a contamination section and lifecycle flags.

A v3 result is a v2 result (``frontierlab.evals.suite_v2``: per-item components, pins, version) with three more keys:

* ``contamination`` — per published item: ``overlap`` (flagged by an n-gram rule against declared training data),
  ``min_k`` (Min-K% Prob) and ``correct``; plus the fresh-item scores, the membership AUC, the fresh gap and
  ``passes`` (see the world-specific module for the rule);
* ``benchmarks`` — the lifecycle entries (:mod:`.lifecycle`) of the benchmarks the result reports, with their flags;
* ``version`` = ``eval-v3.0``.

:func:`compare` applies v2's paired decision rule to every shared component, then states the contamination
verdict of each side. A comparison **passes v3** when it passes v2 *and* neither result fails its contamination
check. Contamination is a property of a model and its data, not of a difference, so it is never "paired away".
"""

from __future__ import annotations

from frontierlab.evals.suite_v2.core import compare as compare_v2
from frontierlab.evals.suite_v2.core import report as report_v2
from frontierlab.evals.suite_v3 import lifecycle

VERSION = "eval-v3.0"


def attach(v2_result: dict, contamination: dict, benchmarks: tuple | list = ()) -> dict:
    out = dict(v2_result)
    out["v2_version"] = v2_result.get("version")
    out["version"] = VERSION
    out["contamination"] = contamination
    out["benchmarks"] = {b: {"status": lifecycle.REGISTRY[b]["status"], "flags": lifecycle.flags(lifecycle.REGISTRY[b])}
                         for b in benchmarks}
    out.setdefault("summary", {})
    out["summary"] = {**out["summary"], "contamination_flagged": contamination["flagged"],
                      "membership_auc": contamination["membership_auc"],
                      "fresh_gap": contamination["fresh_gap"]["gap"], "contamination_passes": contamination["passes"]}
    return out


def compare(base: dict, new: dict, guards: dict | None = None, default_guard: float = 0.02, **kw) -> dict:
    if base.get("version") != VERSION or new.get("version") != VERSION:
        raise ValueError("both results must be Eval v3 results")
    out = compare_v2(base, new, guards, default_guard, **kw)
    out["_contamination"] = {"base_passes": base["contamination"]["passes"], "new_passes": new["contamination"]["passes"],
                             "base_flagged": base["contamination"]["flagged"],
                             "new_flagged": new["contamination"]["flagged"],
                             "base_gap": base["contamination"]["fresh_gap"], "new_gap": new["contamination"]["fresh_gap"]}
    out["_passes_v3"] = out["_passes"] and base["contamination"]["passes"] and new["contamination"]["passes"]
    return out


def report(cmp: dict) -> str:
    c = cmp["_contamination"]
    lines = [report_v2(cmp)]
    for side in ("base", "new"):
        g = c[f"{side}_gap"]
        lines.append(f"contamination ({side}): {c[f'{side}_flagged']} published items flagged by overlap; fresh gap "
                     f"{g['gap']:+.3f} [{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}] -> "
                     f"{'pass' if c[f'{side}_passes'] else 'FAIL'}")
    lines.append(f"Eval v3: {'PASS' if cmp['_passes_v3'] else 'FAIL'}")
    return "\n".join(lines)


__all__ = ["VERSION", "attach", "compare", "report"]
