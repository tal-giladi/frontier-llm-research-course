"""The evaluation-integrity lab (lessons 16.1 and 16.4): how evaluation machinery produces false
success, measured against a defence ladder — and what closes each hole.

    python labs/module-16/evalintegrity/tamper_lab.py                # free CPU, about 1-2 minutes
    python labs/module-16/evalintegrity/tamper_lab.py --part a       # only the isolation profile
    python labs/module-16/evalintegrity/tamper_lab.py --part bcd     # the matrix, ladder, integrity
    python labs/module-16/evalintegrity/tamper_lab.py --make-traces  # regenerate the provided traces
    python labs/module-16/evalintegrity/tamper_lab.py --variant main --print   # container commands

Everything here is a **scripted demonstration**: hand-written fixtures that reproduce published
reward-hacking patterns (MacDiarmid et al. 2025 sec. 2; Baker et al. 2025 sec. 2.1-2.2; METR June
2025; ImpossibleBench) at toy scale, inside a deliberately vulnerable course-owned toy runner
(`toyrunner.py`) that exists only in this folder. Nothing discovered by a policy, nothing run
against the course's real tests or tooling; part D proves the course tree is byte-identical after a
full run.

Part A — the threat model, measured: what a sandboxed candidate can read, write and control.
Part B — the acceptance matrix: every candidate x every verifier level, plus your strict log
         parser (lab TODO 4); nothing is assumed, every cell is a run.
Part C — the defence ladder: which defence newly rejects which demonstration, the measured facts
         that isolation alone and hidden tests alone catch none of it, and the three-way
         distinction between misspecification, visible-test overfitting and evaluation tampering.
Part D — integrity: the course files before and after, and the honest limits of the isolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import demos  # noqa: E402
import toyrunner as T  # noqa: E402
from frontierlab.labkit import load_path  # noqa: E402

TRACES = HERE / "traces"

LEVEL_TITLES = {"exit": "trust the exit code",
                "log": "+ require a complete passing report line",
                "protected": "+ tests and answers are never files in the workspace",
                "channel": "+ values out, compared in the verifier's process, complete execution, no plugins",
                "hardened": "+ hidden and fresh inputs, property checks"}


def _ok(v: dict) -> str:
    return "accept" if v["accepted"] else "REJECT"


def part_a():
    print("(A) the threat model, measured: what a sandboxed candidate can read, write and control")
    p = T.isolation_profile()
    print(f"    workspace (the only files the candidate can read): {p.get('files')}")
    print(f"    environment variables it sees (host secrets stripped): {p.get('env')}")
    print(f"    network: {p.get('network')}")
    print(f"    starting other programs: {p.get('subprocess')}")
    print(f"    writing outside its working directory: {p.get('outside_write')}")
    print("    it controls: its own process (exit code, stdout), the equality of its return values,")
    print("                 any file in its workspace, and any plugin hook the runner supports")
    print("    it never receives: the held-out and fresh inputs' expected values, the verifier's comparison,")
    print("                 or any course file or path in its arguments (part D checks the tree is untouched);")
    print("                 caveat: on some setups the interpreter's own launcher adds PYTHONHOME, which")
    print("                 names the virtual environment: a path outside the workspace the child can learn")


def part_b(lab, seed):
    t0 = time.time()
    rows = demos.run_all(seed=seed)
    print(f"(B) acceptance matrix: {len(rows)} candidates x {len(T.LEVELS)} levels "
          f"(measured {time.time() - t0:.0f}s; every cell is a sandboxed run)")
    head = f"    {'candidate':18s} {'kind':10s} " + "  ".join(f"{l:>9s}" for l in T.LEVELS) + "  strict-log"
    print(head)
    for r in rows:
        strict = lab.strict_log_verdict(r["weak"])
        print(f"    {r['candidate']:18s} {r['kind']:10s} "
              + "  ".join(f"{_ok(r['levels'][l]):>9s}" for l in T.LEVELS) + f"  {'accept' if strict else 'REJECT':>9s}")
    m = lab.acceptance_matrix(rows)
    print(f"    {'level':10s} {'false accepts':>22s} {'tampering accepted':>22s} {'false rejects':>22s}")
    for l in T.LEVELS:
        c = m[l]
        print(f"    {l:10s} {c['false_accepts']:>8d} of {c['bad_total']:<4d} rate {c['false_accept_rate']:.3f}"
              f" {c['tampering_accepts']:>8d} of {c['tampering_total']:<4d}"
              f" {c['false_rejects']:>12d} of {c['correct_total']}   "
              f"(overfit accepted: {c['overfit_accepts']})")
    print("    hardened-level reasons (which defence caught what):")
    for r in rows:
        v = r["levels"]["hardened"]
        if not v["accepted"]:
            print(f"      {r['candidate']:18s} -> {v['reason']}  [{lab.explain_rejection(v)}]")
    return rows


def part_c(lab, rows):
    print("(C) the defence ladder: which defence newly rejects which demonstration (bad candidates only)")
    bad = [r for r in rows if r["kind"] != "correct"]
    prev = None
    for l in T.LEVELS:
        newly = [r["candidate"] for r in bad
                 if (prev is None or r["levels"][prev]["accepted"]) and not r["levels"][l]["accepted"]]
        kept = [r["candidate"] for r in bad if r["levels"][l]["accepted"]]
        print(f"    {l:10s} {LEVEL_TITLES[l]}")
        print(f"               newly rejected: {newly or 'none'}; still accepted: {kept or 'none'}")
        prev = l
    n_tamp = sum(r["kind"] == "tampering" for r in bad)
    print(f"    isolation alone: the exit level already runs isolated, and accepts "
          f"{sum(r['levels']['exit']['accepted'] for r in bad if r['kind'] == 'tampering')} of {n_tamp} tampering demos")
    print(f"    hidden tests alone: the protected level still accepts "
          f"{sum(r['levels']['protected']['accepted'] for r in bad if r['kind'] == 'tampering')} of {n_tamp}")
    print("    the three failure modes this module studies, on this lab's candidates:")
    print("      misspecification  - the reward can be earned without the intended behaviour (16.4's")
    print("                         format-only and length rewards; not on display here)")
    print("      visible-test overfitting - special-casing the inputs the reward checks, harness untouched")
    print("      evaluation tampering - changing what the harness reads (the six demos above; published")
    print("        cases: MacDiarmid et al. 2025 sec. 2, Baker et al. 2025 sec. 2.1-2.2, METR June 2025,")
    print("        ImpossibleBench)")
    for name, description in _BEHAVIOURS.items():
        r = next(x for x in rows if x["candidate"] == name)
        print(f"      {name:18s} -> {lab.classify_behaviour(description)}")
    print("      correct, correct_alt, wrong are not failures of the reward: honest solutions and an")
    print("      honest wrong answer; every verifier's job is to tell them apart.")


_BEHAVIOURS = {
    "exit_early": "The submission called sys.exit(0) before the tests ran.",
    "always_equal": "The submission overrode __eq__ so every assertion passed.",
    "conftest_rewrite": "The runner loaded a workspace conftest.py whose hook rewrote the report to passing.",
    "edit_tests": "The submission rewrote the test file to assert its own wrong outputs.",
    "read_expected": "The submission read the grader's expected answers from a file in its workspace.",
    "print_pass": "The submission printed the passing summary line itself.",
    "hardcode_visible": "The submission returned a lookup table of the visible examples.",
}


def part_d(digest_before):
    same = T.repo_digest(str(HERE.parents[1])) == digest_before
    print(f"(D) integrity: the labs/ tree is byte-identical after the full run: {same} "
          f"({len(digest_before)} files hashed before and after)")
    print("    limits, stated: the sandbox is not a security boundary. The child runs as your OS user;")
    print("    a determined program can write outside its working directory (measured in part A), the")
    print("    network and process guard is in-process and can be undone, and Windows has no memory or")
    print("    CPU rlimit (only the wall-clock timeout). The verdicts therefore never depend on the")
    print("    sandbox: every level above 'exit' treats the child as fully malicious, and compares")
    print("    values in the verifier's own process. For code from capable models, run the levels in a")
    print("    container (--variant main --print).")


def make_traces(lab, rows):
    TRACES.mkdir(exist_ok=True)
    for r in rows:
        r["strict_log"] = lab.strict_log_verdict(r["weak"])
    dest = TRACES / "acceptance-matrix.jsonl"
    dest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8", newline="\n")
    from frontierlab.runcard import hardware
    v = {"records": len(rows), "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
    man = {"description": "Acceptance matrix of the evaluation-integrity lab: 11 scripted candidates "
                          "(2 correct, 2 wrong incl. a hang, 6 tampering demos, 1 visible-example overfit) "
                          "through the toy runner's five verifier levels, plus the strict log parser of "
                          "lab.py. Every cell is a measured sandboxed run on the build machine, not a "
                          "prediction; the demonstrations are hand-written fixtures reproducing published "
                          "reward-hacking patterns (MacDiarmid et al. 2025 sec. 2; Baker et al. 2025 "
                          "sec. 2.1-2.2; METR June 2025; ImpossibleBench) at toy scale.",
           "files": {"acceptance-matrix.jsonl": v},
           "seed": 0, "script": "tamper_lab.py --make-traces", "hardware": hardware(),
           "created": time.strftime("%Y-%m-%d")}
    (TRACES / "manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {dest} ({v['records']} records, sha256 {v['sha256'][:16]}...)")


MAIN = ("docker run --rm --network none --read-only --memory 512m --cpus 1 "
        "--tmpfs /work:rw,size=64m -v {repo}/labs/module-16/evalintegrity:/lab:ro "
        "python:3.12-slim bash -lc 'cp -r /lab /work && cd /work && python tamper_lab.py'")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="abcd")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed of the hardened level's fresh inputs (fixed for reproducible traces)")
    ap.add_argument("--fresh-random", action="store_true",
                    help="also run the hardened level with an unpredictable seed (production setting)")
    ap.add_argument("--make-traces", action="store_true")
    ap.add_argument("--variant", choices=["cpu", "main"], default="cpu")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    if a.variant == "main":
        if a.print:
            print(MAIN.format(repo=os.getcwd()))
            print("# the same matrix inside a container: no network, read-only repo mount,")
            print("# bounded memory and CPU. Not run in this build; part of the Module 16 pilot.")
        return
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    digest_before = T.repo_digest(str(HERE.parents[1]))
    parts = set(a.part)
    rows = None
    if "a" in parts:
        part_a()
    if "b" in parts:
        rows = part_b(lab, a.seed)
    if a.fresh_random and rows is not None:
        changed = []
        for name in ("exit_early", "always_equal", "conftest_rewrite", "edit_tests", "read_expected",
                     "print_pass", "hardcode_visible"):
            v = T.hardened_verdict(demos.CANDIDATES[name]["files"], seed=None)
            if v["accepted"] != rows[[r["candidate"] for r in rows].index(name)]["levels"]["hardened"]["accepted"]:
                changed.append(name)
        print(f"(B2) hardened level with an unpredictable fresh-input seed: verdicts unchanged for "
              f"all 7 tampering demos: {not changed}")
    if "c" in parts and rows is not None:
        part_c(lab, rows)
    if "d" in parts and rows is not None:
        part_d(digest_before)
    if a.make_traces:
        if rows is None:
            rows = demos.run_all(seed=a.seed)
        make_traces(lab, rows)


if __name__ == "__main__":
    main()
