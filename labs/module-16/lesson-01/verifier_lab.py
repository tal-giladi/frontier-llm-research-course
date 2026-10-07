"""Lab 16.1: test the verifiers and the environments before any RL run uses them.

    python labs/module-16/lesson-01/verifier_lab.py              # free CPU, about 1-2 minutes

Part A (coding environment, sandboxed): every verifier on every registered candidate (reference, plausible
bug, visible-pair lookup table) for the six toy specs at three task seeds; your ``confusion`` turns the rows into
false-accept and false-reject rates. Then the hidden-test sweep: how often the plausible bug survives a
verifier with n hidden tests, measured, next to your ``miss_probability`` prediction from each bug's failure
rate on random inputs.

Part B (environment protocol): the reset check on ``CodeEnv`` and ``LeakyCodeEnv``, read by your
``reset_verdict``, and the determinism check.

Part C (program world, in-process): every reward of ``frontierlab.agents.dsl`` on five kinds of answer, over
all 264 tasks: the matrix that lesson 16.4's RL runs will exploit or respect.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

from frontierlab.agents import codeenv as C
from frontierlab.agents import dsl
from frontierlab.agents.env import determinism_check, reset_check
from frontierlab.labkit import load_path

HERE = Path(__file__).resolve().parent


def failure_rate(spec, n: int = 2000, seed: int = 0) -> float:
    """Fraction of random inputs on which the spec's plausible bug differs from the reference (course code)."""
    rng = random.Random(seed)
    ref = C.reference_fn(spec)
    ns: dict = {}
    exec(f"def {spec.signature}:\n{spec.wrong}\n", ns)              # course-owned code
    wrong = ns[spec.name]
    bad = 0
    for _ in range(n):
        a = spec.gen(rng)
        bad += ref(*copy.deepcopy(a)) != wrong(*copy.deepcopy(a))
    return bad / n


def part_a(lab, seeds):
    tasks = [C.make_task(s, seed) for seed in seeds for s in C.SPECS]
    t0 = time.time()
    rep = C.verifier_report(tasks)
    print(f"(A1) {len(tasks)} tasks x {len(C.CANDIDATES)} candidates x {len(C.VERIFIERS)} verifiers "
          f"= {len(rep['rows'])} sandboxed runs, {time.time() - t0:.0f}s")
    conf = lab.confusion(rep["rows"])
    print(f"     {'verifier':9s} {'false accept':>13s} {'loophole accept':>16s} {'false reject':>13s}")
    for v, c in conf.items():
        print(f"     {v:9s} {c['false_accept_rate']:13.3f} {c['loophole_accept_rate']:16.3f} {c['false_reject_rate']:13.3f}")
    print("(A2) the plausible bug against n hidden tests (3 task seeds per n), and the prediction (1 - p)^(n + 3): the verifier also runs the 3 visible pairs")
    print(f"     {'spec':15s} {'p_fail':>7s}  " + "  ".join(f"n={n:<2d} meas/pred" for n in (1, 2, 4, 8)) +
          "   tests for 5% miss")
    for spec in C.SPECS:
        p = failure_rate(spec)
        cells = []
        for n in (1, 2, 4, 8):
            acc = 0
            for seed in seeds:
                task = C.make_task(spec, seed, n_hidden=n)
                files = {**task.data["files"], **C.candidate_files("plausible_bug", task)}
                acc += C.hidden_verify(task, files).passed
            cells.append(f"{acc}/{len(seeds)} {lab.miss_probability(p, n + 3):5.2f}")
        print(f"     {spec.name:15s} {p:7.3f}  " + "  ".join(f"{c:>13s}" for c in cells) +
              f"   {lab.tests_needed(p, 0.05):>6d}")


def part_b(lab):
    task = C.basic_tasks()[0]
    spec = C.SPEC_BY_NAME[task.data["func"]]

    def disturb(env):
        env.step(("write", {"path": "solution.py", "text": C.solution_file(spec, spec.reference)}))

    for cls in (C.CodeEnv, C.LeakyCodeEnv):
        chk = reset_check(cls, task, disturb)
        print(f"(B1) reset check {cls.__name__:13s} {chk} -> {lab.reset_verdict(chk)}")
    chk = reset_check(C.CodeEnv, task, lambda env: env.step(("ls", {})))
    print(f"(B2) reset check with a read-only disturbance -> {lab.reset_verdict(chk)}")
    acts = [("read", {"path": "examples.txt"}), ("write", {"path": "solution.py",
                                                            "text": C.solution_file(spec, spec.reference)}),
            ("run_tests", {})]
    print(f"(B3) determinism check (same seed and actions twice): {determinism_check(lambda: C.CodeEnv('hidden'), task, acts)}")


def part_c():
    tasks = dsl.all_tasks()
    rng = random.Random(0)
    answers = {
        "right rule": lambda t: (t.func.program, True),
        "wrong rule": lambda t: (next(f for f in dsl.ALL_FUNCS if f != t.func).program, True),
        "table": lambda t: (dsl.table_program(t), True),
        "not a program": lambda t: ("x+", True),
        "long, unfinished": lambda t: ("x+3x+3x+3x+3", False),
    }
    names = ["format", "length", "visible", "hidden", "randomised", "property", "gold"]
    print("(C) mean reward over all 264 tasks")
    print(f"     {'answer':17s} " + " ".join(f"{n:>10s}" for n in names))
    for a, fn in answers.items():
        vals = []
        for n in names:
            s = 0.0
            for t in tasks:
                text, fin = fn(t)
                s += dsl.REWARDS[n](text, fin, t, rng)
            vals.append(s / len(tasks))
        print(f"     {a:17s} " + " ".join(f"{v:10.3f}" for v in vals))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args(argv)
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    t0 = time.time()
    part_a(lab, list(range(a.seeds)))
    part_b(lab)
    part_c()
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
