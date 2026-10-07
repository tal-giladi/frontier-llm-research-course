"""The benchmark lifecycle registry of Eval Suite v3 (lesson 18.3). Checked 2026-10-07; volatile by design.

A benchmark is born hard, gets used, gets contaminated or saturated, is audited, and is replaced. Every v3 report
states, for each benchmark it uses, where in that life the benchmark is, from this registry. Entries are
course-written summaries of the cited sources; ``label`` follows the course's evidence labels and ``(S)`` marks a
fact the course could only see in search results or secondary coverage (the page refused automated access).

:func:`flags` turns an entry into the warnings a report must carry; :func:`card` prints one entry.
"""

from __future__ import annotations

CHECKED = "2026-10-07"

REGISTRY: dict[str, dict] = {
    "swe-bench-verified": {
        "released": "2024-08-13", "items": 500,
        "what": "500 SWE-bench tasks screened by 93 professional developers for underspecified issues and tests that "
                "reject valid fixes (S)",
        "status": "retired by OpenAI's evals: 'Why we no longer evaluate SWE-bench Verified', 2026-02-23 (S)",
        "known_issues": ["an audit of 138 hard tasks found at least 59.4% with tests that reject correct fixes (S)",
                         "evidence that frontier models reproduce gold patches (contamination) (S)"],
        "contamination_resistance": "public repositories and fixes, in training corpora since 2023",
        "label": "PUBLICLY DOCUMENTED (company claim; page seen only in search results)",
        "source": "https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/",
        "saturated": False, "public_test_set": True, "audited_flaws": True,
    },
    "swe-bench-pro": {
        "released": "2025-09-21", "items": 1865,
        "what": "1,865 tasks from 41 repositories: public (731, 11 repos), held-out (858, 12 repos, private), "
                "commercial (276, 18 proprietary startup repos)",
        "status": "in use; public leaderboard at labs.scale.com (top score 61.5 +- 3.1 on the public set when checked)",
        "known_issues": ["scores depend on scaffold, turn limit and cost cap; compare only like with like"],
        "contamination_resistance": "public set from copyleft (GPL) repositories; held-out and commercial sets private",
        "label": "PUBLICLY DOCUMENTED (arXiv 2509.16941; leaderboard is a company page)",
        "source": "https://arxiv.org/abs/2509.16941",
        "saturated": False, "public_test_set": True, "audited_flaws": False,
    },
    "hle": {
        "released": "2025-01-24", "items": 2500,
        "what": "Humanity's Last Exam: 2,500 expert-written questions over 100+ subjects, ~14% multimodal, 24% multiple "
                "choice; at release the best model (o3-mini high) scored 13.4% with 80% calibration error",
        "status": "in use; published in Nature (DOI 10.1038/s41586-025-09962-4); arXiv v11 2026-07-28",
        "known_issues": ["FutureHouse (2025-07-23) estimated 29 +- 3.7% of text-only chemistry/biology answers conflict "
                         "with peer-reviewed literature; the HLE team's own review found about 18% of a subset "
                         "problematic"],
        "contamination_resistance": "public questions with a private held-out set (see the paper)",
        "label": "PUBLICLY DOCUMENTED (arXiv 2501.14249; FutureHouse audit)",
        "source": "https://arxiv.org/abs/2501.14249",
        "saturated": False, "public_test_set": True, "audited_flaws": True,
    },
    "frontiermath": {
        "released": "2024-11-07", "items": 350,
        "what": "unpublished, expert-written mathematics problems with automatic answer checking: 300 in Tiers 1-3 and "
                "50 in Tier 4; under 2% solved at release",
        "status": "in use (Epoch AI runs it)",
        "known_issues": ["commissioned and owned by OpenAI, which has access to most problems and solutions; a "
                         "holdout is kept (Epoch AI, 2025-01-23); funding disclosed late"],
        "contamination_resistance": "private problems; access asymmetry between developers",
        "label": "PUBLICLY DOCUMENTED (arXiv 2411.04872; Epoch AI pages)",
        "source": "https://epoch.ai/frontiermath/tiers-1-4/about",
        "saturated": False, "public_test_set": False, "audited_flaws": False,
    },
    "arc-agi-2": {
        "released": "2025-05-17", "items": None,
        "what": "grid-transformation puzzles; a task is kept only if at least two people solved it within two attempts "
                "(407 participants); o3 (medium) scored 3.0% in the paper",
        "status": "in use; ARC Prize 2025 top private-set score 24.0% (S); ARC-AGI-3 (interactive games) announced",
        "known_issues": ["scores below 5% are treated by the authors as not meaningful"],
        "contamination_resistance": "private and semi-private evaluation sets",
        "label": "PUBLICLY DOCUMENTED (arXiv 2505.11831)",
        "source": "https://arxiv.org/abs/2505.11831",
        "saturated": False, "public_test_set": False, "audited_flaws": False,
    },
    "gpqa-diamond": {
        "released": "2023-11-20", "items": 198,
        "what": "198 graduate-level science questions answered correctly by both experts and incorrectly by most "
                "skilled non-experts (GPQA: 448 questions, experts 65%, non-experts with web access 34%)",
        "status": "near saturation: frontier models report above 90% (S, aggregators)",
        "known_issues": ["little headroom; multiple choice, so 25% by chance"],
        "contamination_resistance": "public questions, canary string requested",
        "label": "PUBLICLY DOCUMENTED (arXiv 2311.12022; Epoch AI hub)",
        "source": "https://arxiv.org/abs/2311.12022",
        "saturated": True, "public_test_set": True, "audited_flaws": False,
    },
    "metr-horizon-1.1": {
        "released": "2026-01", "items": 228,
        "what": "METR's time-horizon suite (HCAST, RE-Bench, SWAA): 228 tasks with human baseline times",
        "status": "in use; METR states the suite is nearing saturation for the longest horizons",
        "known_issues": ["error bars about 2x in each direction; horizons differ 40-100x across domains; the 80% "
                         "horizon is not an independent estimate (METR, 2026-01-22)"],
        "contamination_resistance": "mostly private tasks",
        "label": "PUBLICLY DOCUMENTED (METR)",
        "source": "https://metr.org/notes/2026-01-22-time-horizon-limitations/",
        "saturated": False, "public_test_set": False, "audited_flaws": False,
    },
}


def flags(entry: dict) -> list[str]:
    """Warnings a report that uses this benchmark must carry."""
    out = []
    if "retired" in entry["status"]:
        out.append("retired: do not use as a headline result")
    if entry["saturated"]:
        out.append("saturated: differences near the ceiling are not informative")
    if entry["public_test_set"]:
        out.append("public test set: run a contamination check and report a clean or fresh subset")
    if entry["audited_flaws"]:
        out.append("audited label or test errors: report the audited subset or the error rate next to the score")
    return out


def card(name: str) -> str:
    e = REGISTRY[name]
    lines = [f"{name} (released {e['released']}, checked {CHECKED})", f"  what: {e['what']}", f"  status: {e['status']}",
             f"  resistance: {e['contamination_resistance']}", f"  evidence: {e['label']}; {e['source']}"]
    lines += [f"  issue: {i}" for i in e["known_issues"]] + [f"  FLAG: {f}" for f in flags(e)]
    return "\n".join(lines)


__all__ = ["CHECKED", "REGISTRY", "flags", "card"]
