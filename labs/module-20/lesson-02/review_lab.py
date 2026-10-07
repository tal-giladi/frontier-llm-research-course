"""Lab 20.2: review a sample capstone with planted weaknesses, defend it, revise it, log the revision.

    python labs/module-20/lesson-02/review_lab.py                  # step 1: write the sample, check it, list questions
    python labs/module-20/lesson-02/review_lab.py --step rerun     # step 3: the reruns the review requires (simulated)
    python labs/module-20/lesson-02/review_lab.py --step check     # step 5: your defence, revision log and the package
    python labs/module-20/lesson-02/review_lab.py --step reference # the reference revision, for comparison afterwards
    python labs/module-20/lesson-02/review_lab.py --package PATH   # step 6: review your own capstone package

The sample (``frontierlab.capstone.sample``) is a CONSTRUCTED FIXTURE: a fictional learner's GSPO-vs-GRPO capstone
with invented numbers. It is in ``runs/m20/l202/sample``; you write ``defence.md`` and ``revision_log.yaml`` there.
Checks use your ``lab.py`` (``LAB_TARGET=solution`` for the reference).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

from frontierlab.capstone import package as PK
from frontierlab.capstone import review as RV
from frontierlab.capstone import sample as SA
from frontierlab.labkit import load_path

HERE = Path(__file__).resolve().parent
ROOT = Path("runs/m20/l202")


def load_before(pkg: Path) -> list[PK.Problem]:
    return [PK.Problem(**d) for d in json.loads((pkg / "review" / "before.json").read_text())]


def write_review(pkg: Path) -> list[dict]:
    """Check the package as submitted, save the problems and the reviewer questions under <pkg>/review/."""
    probs = PK.check_package(pkg)
    qs = RV.questions(pkg, probs)
    (pkg / "review").mkdir(exist_ok=True)
    (pkg / "review" / "before.json").write_text(json.dumps([p.__dict__ for p in probs], indent=1))
    (pkg / "review" / "questions.json").write_text(json.dumps(qs, indent=1))
    md = ["# Reviewer questions", ""] + [f"### {q['id']}" + (f" ({q['revision']})" if q.get("revision") else "") +
                                         f"\n\n{q['question']}\n" for q in qs]
    (pkg / "review" / "questions.md").write_text("\n".join(md), encoding="utf-8")
    print(f"checker: {len(probs)} problem(s)")
    for p in probs:
        print("  ", p)
    print(f"\n{len(qs)} reviewer questions -> {pkg / 'review' / 'questions.md'}")
    for q in qs:
        print(f"  {q['id']:4s} {q['question']}")
    return qs


def check(lab, pkg: Path) -> bool:
    before = load_before(pkg)
    qs = json.loads((pkg / "review" / "questions.json").read_text())
    after = PK.check_package(pkg)
    ok = True
    print(f"package now: {'SOUND' if not after else f'{len(after)} problem(s)'}")
    for p in after:
        print("  ", p)
    ok &= not after
    d = pkg / "defence.md"
    if d.exists():
        dp = lab.defence_problems(d.read_text(encoding="utf-8"), qs)
        print(f"defence: {'every question answered with evidence' if not dp else dp}")
        ok &= not dp
    else:
        print(f"defence: write {d}")
        ok = False
    r = pkg / "revision_log.yaml"
    if r.exists():
        log = yaml.safe_load(r.read_text(encoding="utf-8")) or []
        rp = lab.revision_log_problems(log, before, after, pkg, (pkg / "contract.md").read_text(encoding="utf-8"))
        print(f"revision log: {'complete' if not rp else rp}")
        ok &= not rp
    else:
        print(f"revision log: write {r}")
        ok = False
    print("\nREVIEW PASSED" if ok else "\nnot yet")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", choices=["write", "rerun", "check", "reference"], default="write")
    ap.add_argument("--package", type=Path, default=None, help="review your own capstone package instead")
    ap.add_argument("--force", action="store_true", help="overwrite an existing sample")
    a = ap.parse_args(argv)
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    if a.package:
        if a.step == "check":
            sys.exit(0 if check(lab, a.package) else 1)
        write_review(a.package)
        return
    pkg = ROOT / "sample"
    if a.step == "write":
        if pkg.exists() and not a.force:
            print(f"{pkg} exists (your edits are kept); use --force to start again")
        else:
            SA.write_sample(pkg, flawed=True)
            print(f"wrote the sample capstone (constructed fixture) to {pkg}\n")
        write_review(pkg)
    elif a.step == "rerun":
        new = SA.apply_reruns(pkg)
        print("rerun outputs written (simulated; constructed numbers):", ", ".join(new))
        print("claim.yaml: GRPO tuning budget now 3 trials; results.json: the extension comparison replaced")
        print("\nnow run the checker again; a rerun can change a result, and a changed result can create a new problem:")
        for p in PK.check_package(pkg):
            print("  ", p)
    elif a.step == "check":
        sys.exit(0 if check(lab, pkg) else 1)
    else:
        ref = ROOT / "reference"
        if ref.exists():
            shutil.rmtree(ref)
        SA.write_sample(ref, flawed=True)
        qs = write_review(ref)
        before = load_before(ref)
        SA.apply_reruns(ref)
        SA.revise_by_hand(ref)
        (ref / "defence.md").write_text(SA.reference_defence(qs), encoding="utf-8")
        (ref / "revision_log.yaml").write_text(yaml.safe_dump(SA.reference_log(before), sort_keys=False, width=1000))
        print("\n== reference revision")
        check(lab, ref)


if __name__ == "__main__":
    main()
