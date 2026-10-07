"""Checking a write-up against its evidence (lesson 19.3).

A research note states its claims twice: in prose for the reader, and in a **claims register** for the checker —
a fenced YAML block whose info string is ``yaml claims``. Each entry names the runs behind the claim, so the claim
can be checked against their run cards before anyone reads the prose::

    - id: c1
      text: "QK-Clip lowers held-out loss against QK-norm at 1.8M parameters."
      metric: val_loss
      arms: {qk-clip: [clip-s0, clip-s1, clip-s2], qk-norm: [norm-s0, norm-s1, norm-s2]}
      changed: [optim.qk_clip, config.qk_norm, optim.qk_norm]   # what the comparison is allowed to change
      axis: tokens                                              # the contract's comparison axis
      value: -0.012                                             # the reported difference
      interval: [-0.020, -0.004]                                # its 95% interval, or null
      checkpoint: {qk-clip: final, qk-norm: final}              # "final", or a step / "best"
      selection_split: val                                      # where checkpoints and hyperparameters were chosen
      scope: "toy preset (1.8M parameters), 300 steps, Data-v0 CPU size"

:func:`lint_claims` returns one :class:`Issue` per problem, with a code a reviewer can act on:

==============  ===================================================================================================
code            raised when
==============  ===================================================================================================
``missing``     a cited run has no run card
``seeds``       an arm rests on fewer than 2 distinct seeds
``uncertainty`` no interval is reported, or the interval contains 0 while the text claims a difference
``budget``      the arms differ on the budget the comparison axis must hold equal (``record.diff_cards``)
``control``     the arms differ in anything else that invalidates the comparison (data, evaluation, a second
                changed variable)
``checkpoint``  a reported number is not from the final checkpoint, or anything was selected on the test split
``scope``       the text generalises beyond the evidence (frontier, every scale, "will grow with scale", ...)
==============  ===================================================================================================

:func:`lint_figure` does the same for a figure specification: one claim per figure, uncertainty shown with at
least 2 seeds per arm, an x-axis that is the contract's comparison axis, arms that end at the same budget on it,
and a caption that says what the bands are and how many seeds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from frontierlab.record import diff_cards, flatten

CODES = ("missing", "seeds", "uncertainty", "budget", "control", "checkpoint", "scope")
COMPARATIVE = re.compile(r"\b(beats?|better|worse|improves?|outperforms?|lower|higher|reduces?|faster|slower|"
                         r"more stable|less stable|wins?)\b", re.I)
OVERREACH = re.compile(r"\b(frontier|all (models|scales|sizes)|every (model|scale|size)|any (model|scale|size)|"
                       r"at scale|always|never fails|in general|state[- ]of[- ]the[- ]art|"
                       r"will (grow|hold|transfer|generali[sz]e)|should (grow|hold|transfer|generali[sz]e))\b", re.I)
BUDGET_KEYS = ("budget.", "args.steps", "args.batch", "args.grad_accum", "measured.")
X_AXES = {"tokens": "tokens", "flops": "flops", "wallclock": "wallclock", "params": "params"}


@dataclass
class Issue:
    code: str
    where: str          # claim id or "figure"
    message: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.where}: {self.message}"


def parse_claims(markdown: str) -> list[dict]:
    """The claims register of a write-up: every ```yaml claims block, concatenated."""
    out = []
    for block in re.findall(r"```ya?ml[ \t]+claims[ \t]*\r?\n(.*?)```", markdown, flags=re.S):
        items = yaml.safe_load(block) or []
        out.extend(items)
    return out


def load_cards(folder) -> dict:
    """{run name: card} for every ``*.yaml`` file (or ``*/run_card.yaml``) under ``folder``."""
    folder, cards = Path(folder), {}
    for p in sorted(folder.glob("*.yaml")) + sorted(folder.glob("*/run_card.yaml")):
        card = yaml.safe_load(p.read_text(encoding="utf-8"))
        cards[card.get("run") or p.stem] = card
    return cards


def _seed(card: dict):
    a = card.get("args", {})
    return a.get("seed"), a.get("data_seed")


def _params(card: dict) -> float:
    return float(flatten(card).get("params.total", 0) or 0)


def lint_claims(claims: list[dict], cards: dict) -> list[Issue]:
    """Problems with each registered claim, checked against the run cards it cites (codes in the module docstring)."""
    issues = []
    for c in claims:
        cid = str(c.get("id", "?"))
        arms = c.get("arms") or {}
        runs = {arm: list(rs or []) for arm, rs in arms.items()}
        missing = [r for rs in runs.values() for r in rs if r not in cards]
        if missing:
            issues.append(Issue("missing", cid, f"no run card for {', '.join(missing)}"))
        have = {arm: [cards[r] for r in rs if r in cards] for arm, rs in runs.items()}
        for arm, cs in have.items():
            seeds = {_seed(x) for x in cs}
            if len(seeds) < 2:
                issues.append(Issue("seeds", cid, f"arm '{arm}' rests on {len(seeds)} seed(s); "
                                    "a difference needs at least 2 per arm to compare with the seed noise"))
        interval = c.get("interval")
        text = str(c.get("text", ""))
        if not interval:
            issues.append(Issue("uncertainty", cid, "no interval reported for the difference"))
        elif interval[0] <= 0 <= interval[1] and COMPARATIVE.search(text):
            issues.append(Issue("uncertainty", cid, f"the interval [{interval[0]}, {interval[1]}] contains 0 "
                                "but the text claims a difference"))
        names = list(have)
        if len(names) >= 2 and all(have[n] for n in names):
            ref = have[names[0]][0]
            seen = set()
            for other in names[1:]:
                for f in diff_cards(ref, have[other][0], changed=tuple(c.get("changed") or ()),
                                    axis=c.get("axis", "tokens"), seeds_are_replicates=True):
                    if f.severity != "invalidates" or f.key in seen:
                        continue
                    seen.add(f.key)
                    code = "budget" if f.key.startswith(BUDGET_KEYS) else "control"
                    issues.append(Issue(code, cid, f"{names[0]} vs {other}: {f.key} {f.a!r} -> {f.b!r} ({f.reason})"))
        ck = c.get("checkpoint") or {}
        bad = {arm: v for arm, v in ck.items() if str(v) != "final"}
        if bad:
            issues.append(Issue("checkpoint", cid, f"not the final checkpoint: {bad}; a pre-stated selection rule "
                                "applied to every arm on the validation split is the only alternative"))
        if str(c.get("selection_split", "val")).lower() == "test":
            issues.append(Issue("checkpoint", cid, "checkpoints or hyperparameters were selected on the test split"))
        m = OVERREACH.search(text + " " + str(c.get("scope", "")))
        if m:
            biggest = max((_params(x) for cs in have.values() for x in cs), default=0.0)
            issues.append(Issue("scope", cid, f"'{m.group(0)}' generalises beyond the evidence "
                                f"(largest model behind this claim: {biggest:,.0f} parameters)"))
    return issues


def lint_writeup(markdown_path, cards_folder) -> list[Issue]:
    md = Path(markdown_path).read_text(encoding="utf-8")
    claims = parse_claims(md)
    if not claims:
        return [Issue("missing", "write-up", "no ```yaml claims block: register every claim with its runs")]
    return lint_claims(claims, load_cards(cards_folder))


def lint_figure(spec: dict) -> list[Issue]:
    """Problems with a figure specification::

        claim: one sentence            # a list of several claims is a problem
        x: tokens | flops | wallclock | params | steps | lr | ...
        contract_axis: tokens          # the experiment contract's comparison axis
        uncertainty: "min-max over 3 seeds" | "95% CI" | none
        arms: {name: {n_seeds: 3, tokens_per_step: 2048, x_end: 6.1e5}}
        caption: "..."
    """
    issues = []
    claim = spec.get("claim")
    if not claim or (isinstance(claim, list) and len(claim) != 1):
        issues.append(Issue("scope", "figure", "one figure should answer one claim; split it or state the one claim"))
    arms = spec.get("arms") or {}
    unc = str(spec.get("uncertainty") or "none").strip().lower()
    if unc in ("none", "", "no"):
        issues.append(Issue("uncertainty", "figure", "no uncertainty shown (bands or bars over seeds or items)"))
    few = [a for a, v in arms.items() if int(v.get("n_seeds", 0)) < 2]
    if few:
        issues.append(Issue("seeds", "figure", f"arms with fewer than 2 seeds: {', '.join(few)}"))
    x, axis = str(spec.get("x", "")), str(spec.get("contract_axis", ""))
    tps = {v.get("tokens_per_step") for v in arms.values() if v.get("tokens_per_step") is not None}
    if x == "steps" and len(tps) > 1:
        issues.append(Issue("budget", "figure", "x is optimizer steps but the arms use different tokens per step; "
                            "plot against the comparison axis"))
    if x in X_AXES and axis in X_AXES and x != axis:
        issues.append(Issue("budget", "figure", f"x is '{x}' but the contract compares at equal '{axis}'"))
    ends = [float(v["x_end"]) for v in arms.values() if v.get("x_end") is not None]
    if len(ends) >= 2 and (max(ends) - min(ends)) / max(abs(max(ends)), 1e-12) > 0.01:
        issues.append(Issue("budget", "figure", f"arms end at different budgets on the x-axis ({min(ends):g} to {max(ends):g})"))
    cap = str(spec.get("caption", "")).lower()
    if not re.search(r"seed|n\s*=", cap) or not re.search(r"interval|ci\b|range|std|standard|band|bar", cap):
        issues.append(Issue("uncertainty", "figure", "the caption must say what the bands or bars are and over how many seeds"))
    return issues
