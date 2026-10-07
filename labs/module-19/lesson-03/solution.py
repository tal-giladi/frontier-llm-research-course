"""Reference solution for lab 19.3."""

from __future__ import annotations

import re

OVERREACH = re.compile(r"\b(frontier|all (models|scales|sizes)|every (model|scale|size)|any (model|scale|size)|"
                       r"at scale|always|in general|state[- ]of[- ]the[- ]art|"
                       r"(will|should) (grow|hold|transfer|generali[sz]e))\b", re.I)

# The review of ../flawed/writeup.md, one entry per problem: the four planted ones first, then what else was found.
REVIEW = [
    {"problem": "missing seeds", "where": "Setup; claims register arms",
     "evidence": "clip-s0 and norm-s0 both have args.seed 0: one run per arm, no interval",
     "request": "three or more seeds per arm, shared by seed across arms, and the paired interval over seeds"},
    {"problem": "unmatched budget", "where": "Setup, QK-Clip arm",
     "evidence": "clip-s0 budget.tokens 819200 against norm-s0 614400 (400 vs 300 steps) on an equal-tokens comparison",
     "request": "both arms at 300 steps, or both continued to 400 with the same schedule length"},
    {"problem": "cherry-picked checkpoint", "where": "Setup, evaluation; claims register checkpoint and selection_split",
     "evidence": "best of 8 evaluations on the test split for QK-Clip, final evaluation for QK-norm",
     "request": "final checkpoints of every run on the validation split; the test split untouched"},
    {"problem": "claim beyond evidence", "where": "Summary",
     "evidence": "'should transfer to frontier models' from 1,836,416-parameter runs",
     "request": "scope the claim to the toy preset, 300 steps and lr 1e-2, or show a ladder"},
    {"problem": "figure", "where": "Figure 1",
     "evidence": "two claims, x in steps with arms ending at 300 and 400, no uncertainty, caption without seeds",
     "request": "one claim, final loss per seed at equal tokens with the paired interval, caption stating n"},
]


def seed_issues(arms: dict) -> list[str]:
    out = []
    for arm, cards in arms.items():
        seeds = {(c.get("args", {}).get("seed"), c.get("args", {}).get("data_seed")) for c in cards}
        if len(seeds) < 2:
            out.append(arm)
    return out


def checkpoint_issues(claim: dict) -> list[str]:
    out = [f"{arm}: {v}" for arm, v in (claim.get("checkpoint") or {}).items() if str(v) != "final"]
    if str(claim.get("selection_split", "val")).lower() == "test":
        out.append("selected on the test split")
    return out


def scope_issue(text: str):
    m = OVERREACH.search(text)
    return m.group(0) if m else None


def figure_issues(spec: dict) -> set[str]:
    codes = set()
    claim = spec.get("claim")
    if not claim or (isinstance(claim, list) and len(claim) != 1):
        codes.add("scope")
    arms = spec.get("arms") or {}
    if str(spec.get("uncertainty") or "none").strip().lower() in ("none", "", "no"):
        codes.add("uncertainty")
    if any(int(v.get("n_seeds", 0)) < 2 for v in arms.values()):
        codes.add("seeds")
    tps = {v.get("tokens_per_step") for v in arms.values() if v.get("tokens_per_step") is not None}
    axes = ("tokens", "flops", "wallclock", "params")
    x, axis = str(spec.get("x", "")), str(spec.get("contract_axis", ""))
    if (x == "steps" and len(tps) > 1) or (x in axes and axis in axes and x != axis):
        codes.add("budget")
    ends = [float(v["x_end"]) for v in arms.values() if v.get("x_end") is not None]
    if len(ends) >= 2 and (max(ends) - min(ends)) / max(abs(max(ends)), 1e-12) > 0.01:
        codes.add("budget")
    cap = str(spec.get("caption", "")).lower()
    if not re.search(r"seed|n\s*=", cap) or not re.search(r"interval|ci\b|range|std|standard|band|bar", cap):
        codes.add("uncertainty")
    return codes
