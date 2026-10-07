"""Lab 19.3 — writing and reviewing results. Fill in the TODOs; run `pytest labs/module-19/lesson-03` to check.
``review_lab.py`` uses these functions and your REVIEW.
"""

from __future__ import annotations

# TODO 5: your review of flawed/writeup.md, one dict per problem, using review-template.md as the guide:
#   {"problem": ..., "where": the section or field, "evidence": the card field or sentence that shows it,
#    "request": what the author must run or change}
# "problem" for the planted ones must be exactly one of: "missing seeds", "unmatched budget",
# "cherry-picked checkpoint", "claim beyond evidence". Add any further problems you find with your own label.
REVIEW: list[dict] = []


def seed_issues(arms: dict) -> list[str]:
    """``arms``: {arm name: [run card dicts]}. The arms whose cards contain fewer than 2 distinct seeds, where a
    seed is the pair (args.seed, args.data_seed). In the order of ``arms``."""
    raise NotImplementedError("TODO 1: does each arm rest on at least two seeds?")


def checkpoint_issues(claim: dict) -> list[str]:
    """One string per problem: "<arm>: <value>" for every arm whose ``claim["checkpoint"]`` value is not "final"
    (in the dict's order), then "selected on the test split" if ``claim["selection_split"]`` is "test"."""
    raise NotImplementedError("TODO 2: was any number picked after looking at it?")


def scope_issue(text: str):
    """The first phrase in ``text`` that generalises beyond a small-scale experiment (for example "frontier",
    "all models", "at scale", "always", "should transfer", "will grow"), or None."""
    raise NotImplementedError("TODO 3: does the claim say more than its evidence?")


def figure_issues(spec: dict) -> set[str]:
    """The set of issue codes ``frontierlab.research.writeup.lint_figure`` would raise for this figure spec
    ("scope", "uncertainty", "seeds", "budget"). Read lint_figure's docstring for the rules; write them yourself."""
    raise NotImplementedError("TODO 4: does the figure answer one question honestly?")
