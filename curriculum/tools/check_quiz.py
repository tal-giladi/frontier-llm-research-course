"""Check quiz YAML files against the Academy rules (binding guide section 5).

    python curriculum/tools/check_quiz.py lessons/module-01/*.quiz.yaml assessments/*.quiz.yaml
Exit code 1 if any file breaks a rule. Warnings (style) do not fail.
"""
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

BANNED = re.compile(r"\b(all|none) of the above\b", re.I)


def check(path: Path, is_module: bool):
    errors, warns = [], []
    try:
        qs = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return [f"YAML does not parse: {e}"], []
    lo, hi = (8, 10) if is_module else (3, 5)
    if not isinstance(qs, list) or not lo <= len(qs) <= hi:
        errors.append(f"{len(qs) if isinstance(qs, list) else 0} questions, need {lo}-{hi}")
        return errors, warns
    ids = [q.get("id") for q in qs]
    if len(set(ids)) != len(ids):
        errors.append("duplicate ids")
    pos = Counter()
    for q in qs:
        qid = q.get("id")
        opts = q.get("options", [])
        if set(q) - {"id", "question", "options", "correct", "explanation"}:
            errors.append(f"{qid}: unexpected fields {set(q) - {'id', 'question', 'options', 'correct', 'explanation'}}")
        if not all(isinstance(o, str) and o.strip() for o in opts) or len(opts) != 4 or len({str(o).strip() for o in opts}) != 4:
            errors.append(f"{qid}: need 4 distinct options")
        c = q.get("correct")
        if c not in (0, 1, 2, 3):
            errors.append(f"{qid}: correct must be 0-3")
            continue
        if not str(q.get("explanation", "")).strip():
            errors.append(f"{qid}: missing explanation")
        if any(BANNED.search(str(o)) for o in opts):
            errors.append(f"{qid}: all/none of the above")
        pos[c] += 1
        lens = [len(str(o)) for o in opts]
        others = max(l for i, l in enumerate(lens) if i != c)
        if lens[c] > 1.15 * others:
            warns.append(f"{qid}: correct option is noticeably longest ({lens[c]} vs {others})")
    if len(qs) >= 4 and max(pos.values()) > (len(qs) + 1) // 2:
        warns.append(f"correct positions clustered: {dict(pos)}")
    return errors, warns


def main(paths):
    bad = False
    for p in map(Path, paths):
        errors, warns = check(p, "assessments" in p.parts)
        for e in errors:
            print(f"ERROR {p}: {e}")
        for w in warns:
            print(f"warn  {p}: {w}")
        bad |= bool(errors)
    print("quiz check:", "FAILED" if bad else "ok", f"({len(paths)} files)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
