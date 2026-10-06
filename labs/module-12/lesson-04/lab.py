"""Lab 12.4 — Eval Suite v2. Fill in the TODOs; run `pytest labs/module-12/lesson-04` to check."""

from __future__ import annotations

import math
import re


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct: 1 - C(n-c, k) / C(n, k) (Chen et al. 2021, Eq. 1).

    Compute it as a running product, not with factorials (n can be 200), and return 1.0 when n - c < k.
    """
    raise NotImplementedError("TODO 1: unbiased pass@k")


def follows_tag(text: str, finished: bool, tag: str, digits: int = 2) -> bool:
    """Format-only instruction check for the toy world (the number itself is not checked):

    an unfinished response never follows; "." -> digits only; "P" -> exactly digits+1 digits;
    "Q" -> '#', digits, '#'; "E" -> digits then '!'.
    """
    raise NotImplementedError("TODO 2: instruction check")


def verdict(kind: str, lo: float, hi: float, guard: float) -> str:
    """Eval v2's pre-stated rule for one component, given the 95% interval [lo, hi] of (new - base):

    task: "improved" if lo > 0, "worse" if hi < 0, else "no clear change".
    retention / instruction: "regressed" if lo < -guard, else "held".
    """
    raise NotImplementedError("TODO 3: the retention guard")


def prompt_and_instruction_level(flags: list[list[bool]]) -> tuple[float, float]:
    """IFEval's two accuracies from per-prompt lists of per-instruction results:
    prompt-level = fraction of prompts with every instruction followed; instruction-level = fraction of
    all instructions followed."""
    raise NotImplementedError("TODO 4: prompt- and instruction-level accuracy")


def loose_variants(response: str) -> list[str]:
    """IFEval's 8 loose variants, in this order: [r, first line removed, last line removed, both removed]
    (lines split on "\\n", rejoined with "\\n", stripped; the first one is the response unchanged), then
    the same four with every '*' removed."""
    raise NotImplementedError("TODO 5: loose variants")
