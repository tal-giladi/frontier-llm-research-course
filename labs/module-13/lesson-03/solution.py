"""Lab 13.3 — Specification-driven alignment. Reference solution."""

from __future__ import annotations

import re

import numpy as np

REFUSAL = "nil"


def spec_check(a, tag, text, finished, digits=2):
    if not finished:
        return False, "unfinished"
    if a >= 90:
        return (True, "ok") if text == REFUSAL else (False, "answered_restricted")
    if text == REFUSAL:
        return False, "over_refusal"
    pattern = {".": r"\d+", "P": r"\d{%d}" % (digits + 1), "Q": r"#\d+#", "E": r"\d+!"}[tag]
    return (True, "ok") if re.fullmatch(pattern, text) else (False, "format")


def judge_errors(probs, truth, threshold=0.5):
    pred = np.asarray(probs) > threshold
    truth = np.asarray(truth).astype(bool)
    bad, good = ~truth, truth
    return {"false_accept": float((pred & bad).sum() / bad.sum()) if bad.any() else 0.0,
            "false_reject": float((~pred & good).sum() / good.sum()) if good.any() else 0.0,
            "accuracy": float((pred == truth).mean())}


def pair_from_scores(scores, margin=0.0):
    s = np.asarray(scores, dtype=float)
    hi, lo = int(np.argmax(s)), int(np.argmin(s))
    if s[hi] - s[lo] <= margin:
        return None
    return hi, lo


def combined_score(judge_prob, number_right, weight=0.5):
    return float(judge_prob) + (weight if number_right else 0.0)
