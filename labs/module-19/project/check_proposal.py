"""Module 19 project: check a capstone proposal before any GPU time is spent, and total its rubric self-score.

    python labs/module-19/project/check_proposal.py runs/m19/project/proposal.yaml
    python labs/module-19/project/check_proposal.py runs/m19/project/proposal.yaml --results-on 2026-10-20
    python labs/module-19/project/check_proposal.py runs/m19/project/proposal.yaml --rubric runs/m19/project/rubric.yaml

Prints every problem ``frontierlab.research.contract.validate_proposal`` finds, then the design's numbers: the minimum
detectable effect of the stated seeds and noise floor, the power at the expected effect, and the capstone claim's
course-lab cost for comparison. With ``--rubric`` it totals a YAML of {criterion: 0|1|2} for the seven criteria of
templates/experiment-rubric.md (keys: question, controls, axis_budget, correctness, uncertainty, conclusion, limits).
Exit code 1 if the proposal has problems.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from frontierlab.research.contract import RUBRIC, mde, rubric_total, validate_proposal
from frontierlab.research.proposals import CAPSTONE_CLAIMS, power


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("proposal", type=Path)
    ap.add_argument("--results-on", default=None, help="ISO date of your first result, if any")
    ap.add_argument("--rubric", type=Path, default=None)
    a = ap.parse_args(argv)
    c = yaml.safe_load(a.proposal.read_text(encoding="utf-8"))
    probs = validate_proposal(c, results_on=a.results_on)
    print(f"{a.proposal}: " + ("no problems" if not probs else f"{len(probs)} problem(s)"))
    for p in probs:
        print(f"  - {p}")
    try:
        m = c["metrics"]
        need = mde(c)
        pw = power(float(m["expected_effect"]), float(m["noise_floor"]), int(m["n_seeds"]))
        print(f"design: noise floor {float(m['noise_floor']):g}, {int(m['n_seeds'])} seeds per arm -> minimum detectable "
              f"effect {need:.4g} (80% power); power at the expected effect {float(m['expected_effect']):g}: {pw:.2f}")
    except (KeyError, TypeError, ValueError):
        print("design: noise floor, seeds or expected effect missing; no power calculation")
    claim = CAPSTONE_CLAIMS.get(c.get("claim_id"))
    if claim:
        lo, hi = claim["lab_cost_gpu_h"]
        cost = f"{lo:g}" if lo == hi else f"{lo:g}-{hi:g}"
        print(f"claim: {claim['title']} ({claim['source']}); course lab cost {cost} GPU-h ({claim['cost_source']}); "
              f"your budget {c.get('budget', {}).get('gpu_hours')} GPU-h {c.get('budget', {}).get('label', '')}")
        print(f"known proxy risk: {claim['proxy_risk']}")
    if a.rubric:
        scores = yaml.safe_load(a.rubric.read_text(encoding="utf-8"))
        r = rubric_total(scores)
        for k, name in RUBRIC:
            print(f"  {name:36s} {scores[k]}")
        print(f"rubric: {r['total']} of {r['max']}" + (" - pass" if r["passed"] else
              f" - not yet (need 10 with no zero; zeros: {r['zeros']})"))
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
