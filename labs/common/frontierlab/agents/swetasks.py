"""Software-engineering task synthesis and group-level splits, at toy scale (lesson 16.2).

SWE-smith (Yang et al. 2025, section 2.1) turns a working repository into many task instances by **breaking**
it: it modifies code (LM rewrites, procedural AST modifications, combined patches, reverted pull requests),
runs the repository's tests, and keeps a candidate only if it makes at least one previously passing test fail.
The task is then "make the failing tests pass again", and the original code is the reference fix. This module
does the same on four toy repositories of small Python functions, with five procedural AST modifications
(the **task families**):

=================  ===============================================
family             change
=================  ===============================================
``flip_compare``   ``<`` <-> ``<=``, ``>`` <-> ``>=``, ``==`` <-> ``!=``
``shift_const``    an integer literal n -> n + 1 or n - 1
``swap_binop``     ``+`` <-> ``-``, ``*`` -> ``+``
``swap_minmax``    a call to ``min`` <-> ``max``
``drop_not``       ``not x`` -> ``x``
=================  ===============================================

Every candidate is validated by running the function's tests on it in the sandbox (one subprocess per
repository); a candidate that crashes on import, times out, or passes every test is discarded.

**Why the split unit matters.** Tasks generated from one function share almost all their code with each other
and have the *same* fix. A random split of instances puts siblings on both sides, so a solver that memorises
fixes looks good on the held-out part. :func:`retrieval_solver` is such a solver: it proposes the fix of the
most similar training task. :func:`search_solver` is a solver that does not use the training set at all: it
tries every single inverse modification and keeps one that passes the visible tests. Comparing the two across
split units (instance, function, repository, family) shows which held-out scores measure generalisation.

:func:`swesmith_overlap` measures the same thing on the real SWE-smith dataset from instance metadata: the
share of held-out instances whose patch touches a file, or whose failing tests appear, in a training instance.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import random
import re
from dataclasses import dataclass

from frontierlab.agents.sandbox import run_python

# --------------------------------------------------------------------------- toy repositories

REPOS: dict[str, dict[str, tuple[str, object]]] = {
    "textkit": {
        "count_vowels": ("def count_vowels(s):\n    return sum(1 for ch in s.lower() if ch in 'aeiou')\n",
                         lambda r: ("".join(r.choice("abeiOUxy ") for _ in range(r.randint(0, 9))),)),
        "is_palindrome": ("def is_palindrome(s):\n    t = [c for c in s.lower() if c.isalpha()]\n"
                          "    for i in range(len(t) // 2):\n        if t[i] != t[len(t) - 1 - i]:\n"
                          "            return False\n    return True\n",
                          lambda r: (r.choice(["abba", "abc", "Aba", "racecar", "ab", "", "a b a", "xyzzyx"]),)),
        "first_repeat": ("def first_repeat(s):\n    seen = set()\n    for ch in s:\n        if ch in seen:\n"
                         "            return ch\n        seen.add(ch)\n    return None\n",
                         lambda r: ("".join(r.choice("abcd") for _ in range(r.randint(0, 6))),)),
        "word_lengths": ("def word_lengths(s):\n    return [len(w) for w in s.split() if len(w) > 0]\n",
                         lambda r: (" ".join("x" * r.randint(1, 5) for _ in range(r.randint(0, 4))),)),
    },
    "listkit": {
        "running_max": ("def running_max(xs):\n    out = []\n    for x in xs:\n"
                        "        out.append(x if not out else max(out[-1], x))\n    return out\n",
                        lambda r: ([r.randint(-5, 9) for _ in range(r.randint(1, 6))],)),
        "second_largest": ("def second_largest(xs):\n    s = sorted(set(xs))\n"
                           "    return s[-2] if len(s) >= 2 else None\n",
                           lambda r: ([r.randint(0, 6) for _ in range(r.randint(0, 6))],)),
        "chunk": ("def chunk(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs), n)]\n",
                  lambda r: (list(range(r.randint(0, 7))), r.randint(1, 3))),
        "count_less": ("def count_less(xs, t):\n    return sum(1 for x in xs if x < t)\n",
                       lambda r: ([r.randint(0, 9) for _ in range(r.randint(0, 6))], r.randint(0, 9))),
    },
    "mathkit": {
        "clamp": ("def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
                  lambda r: (r.randint(-9, 12), 0, 5)),
        "digit_sum": ("def digit_sum(n):\n    return sum(int(c) for c in str(abs(n)))\n",
                      lambda r: (r.randint(-500, 500),)),
        "is_even_len": ("def is_even_len(xs):\n    return not len(xs) % 2\n",
                        lambda r: (list(range(r.randint(0, 5))),)),
        "triangle": ("def triangle(n):\n    return n * (n + 1) // 2\n", lambda r: (r.randint(0, 20),)),
    },
    "datekit": {
        "is_leap": ("def is_leap(y):\n    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)\n",
                    lambda r: (r.choice([1900, 2000, 2023, 2024, 2100, 1996, 2001, 2400]),)),
        "days_in_month": ("def days_in_month(m):\n    if m == 2:\n        return 28\n"
                          "    return 30 if m in (4, 6, 9, 11) else 31\n", lambda r: (r.randint(1, 12),)),
        "clock_add": ("def clock_add(h, d):\n    return (h + d) % 12\n", lambda r: (r.randint(0, 11), r.randint(0, 30))),
        "quarter": ("def quarter(m):\n    return (m - 1) // 3 + 1\n", lambda r: (r.randint(1, 12),)),
    },
}

FAMILIES = ("flip_compare", "shift_const", "swap_binop", "swap_minmax", "drop_not")
_CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add}


def _sites(tree: ast.AST, family: str) -> list:
    out = []
    for node in ast.walk(tree):
        if family == "flip_compare" and isinstance(node, ast.Compare):
            out += [(node, i) for i, op in enumerate(node.ops) if type(op) in _CMP]
        elif family == "shift_const" and isinstance(node, ast.Constant) and type(node.value) is int:
            out += [(node, +1), (node, -1)]
        elif family == "swap_binop" and isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            out.append((node, None))
        elif family == "swap_minmax" and isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("min", "max"):
            out.append((node, None))
        elif family == "drop_not" and isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            out.append((node, None))
    return out


def mutate(source: str, family: str) -> list[str]:
    """Every single-site modification of ``family`` applied to ``source`` (one new source per site)."""
    tree = ast.parse(source)
    n = len(_sites(tree, family))
    out = []
    for k in range(n):
        t = copy.deepcopy(tree)
        node, i = _sites(t, family)[k]
        if family == "flip_compare":
            node.ops[i] = _CMP[type(node.ops[i])]()
        elif family == "shift_const":
            node.value = node.value + i
        elif family == "swap_binop":
            node.op = _BIN[type(node.op)]()
        elif family == "swap_minmax":
            node.func.id = "max" if node.func.id == "min" else "min"
        elif family == "drop_not":
            parent = _parent_of(t, node)
            _replace_child(parent, node, node.operand)
        new = ast.unparse(t) + "\n"
        if new != ast.unparse(tree) + "\n":
            out.append(new)
    return out


def _parent_of(tree, target):
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            if child is target:
                return node
    raise ValueError("no parent")


def _replace_child(parent, old, new):
    for field, value in ast.iter_fields(parent):
        if value is old:
            setattr(parent, field, new)
            return
        if isinstance(value, list):
            for j, v in enumerate(value):
                if v is old:
                    value[j] = new
                    return
    raise ValueError("child not found")


# --------------------------------------------------------------------------- tests and validation

def make_tests(func: str, seed: int = 0, n: int = 12) -> list[list]:
    """Distinct input tuples for ``func`` (as JSON-able lists), from its generator."""
    repo = repo_of(func)
    gen = REPOS[repo][func][1]
    rng = random.Random(f"{func}-{seed}")
    seen, out = set(), []
    for _ in range(200):
        a = list(gen(rng))
        k = json.dumps(a)
        if k not in seen:
            seen.add(k)
            out.append(a)
        if len(out) >= n:
            break
    return out


def repo_of(func: str) -> str:
    for r, fs in REPOS.items():
        if func in fs:
            return r
    raise KeyError(func)


BATCH_HARNESS = '''import json, sys
jobs = json.loads(sys.stdin.read())
out = []
for job in jobs:
    ns = {}
    try:
        exec(job["source"], ns)
        f = ns[job["func"]]
        res = []
        for a in job["inputs"]:
            try:
                res.append({"ok": True, "value": f(*a)})
            except Exception as e:  # noqa: BLE001
                res.append({"ok": False, "error": type(e).__name__})
        out.append({"ok": True, "results": res})
    except Exception as e:  # noqa: BLE001
        out.append({"ok": False, "error": type(e).__name__})
with open("results.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh)
'''


def run_batch(jobs: list[dict], timeout: float = 20.0) -> list[dict] | None:
    """Run many (source, func, inputs) jobs in one sandboxed subprocess; None if the batch timed out."""
    res = run_python(files={"_batch.py": BATCH_HARNESS}, script="_batch.py", stdin=json.dumps(jobs),
                     timeout=timeout, collect=["results.json"], max_output=50_000_000)
    raw = res.files.get("results.json")
    return None if raw is None else json.loads(raw)


@dataclass(frozen=True)
class SweTask:
    id: str
    repo: str
    func: str
    family: str
    buggy: str
    fix: str
    failing: tuple[int, ...]            # indexes of the tests the bug breaks

    @property
    def key(self) -> dict:
        return {"instance": self.id, "function": self.func, "repo": self.repo, "family": self.family}


def synthesise(seed: int = 0, n_tests: int = 12) -> tuple[list[SweTask], dict]:
    """All validated tasks over every repository and family, plus counts of what was discarded and why."""
    tasks, stats = [], {"candidates": 0, "kept": 0, "no_failing_test": 0, "crashed": 0, "timeout_batches": 0}
    for repo, funcs in REPOS.items():
        jobs, meta = [], []
        for func, (src, _gen) in funcs.items():
            inputs = make_tests(func, seed, n_tests)
            jobs.append({"source": src, "func": func, "inputs": inputs})
            meta.append((func, "reference", src))
            for fam in FAMILIES:
                for m in mutate(src, fam):
                    jobs.append({"source": m, "func": func, "inputs": inputs})
                    meta.append((func, fam, m))
        out = run_batch(jobs)
        if out is None:
            stats["timeout_batches"] += 1
            continue
        ref = {}
        for (func, fam, src), r in zip(meta, out):
            if fam == "reference":
                ref[func] = r["results"]
        for (func, fam, src), r in zip(meta, out):
            if fam == "reference":
                continue
            stats["candidates"] += 1
            if not r["ok"]:
                stats["crashed"] += 1
                continue
            failing = tuple(i for i, (a, b) in enumerate(zip(r["results"], ref[func])) if a != b)
            if not failing:
                stats["no_failing_test"] += 1
                continue
            h = hashlib.sha1(src.encode()).hexdigest()[:8]
            tasks.append(SweTask(f"{repo}.{func}.{fam}__{h}", repo, func, fam, src, funcs[func][0], failing))
            stats["kept"] += 1
    return tasks, stats


# --------------------------------------------------------------------------- splits and solvers

def split(tasks: list[SweTask], by: str, held_frac: float = 0.25, seed: int = 0) -> tuple[list, list]:
    """Hold out a fraction of the *groups* named by ``by`` (instance, function, repo or family)."""
    keys = sorted({t.key[by] for t in tasks})
    rng = random.Random(f"{by}-{seed}")
    rng.shuffle(keys)
    held = set(keys[:max(1, round(held_frac * len(keys)))])
    return [t for t in tasks if t.key[by] not in held], [t for t in tasks if t.key[by] in held]


_TOK = re.compile(r"\w+|[^\w\s]")


def jaccard(a: str, b: str) -> float:
    x, y = set(_TOK.findall(a)), set(_TOK.findall(b))
    return len(x & y) / max(1, len(x | y))


def retrieval_solver(task: SweTask, train: list[SweTask]) -> str:
    """Propose the fix of the training task whose buggy code is most similar (ties: first)."""
    best = max(train, key=lambda t: jaccard(t.buggy, task.buggy))
    return best.fix


def search_solver(task: SweTask) -> list[str]:
    """Candidate repairs of ``task.buggy`` that do not depend on any training data: every single modification
    of every family (the inverse of a modification is again a modification of some family). The caller keeps the
    first candidate that passes the visible tests."""
    out = []
    for fam in FAMILIES:
        out += mutate(task.buggy, fam)
    return out


def solve_and_score(tasks_held: list[SweTask], train: list[SweTask], solver: str, seed: int = 0,
                    n_visible: int = 4, n_tests: int = 12) -> dict:
    """Success rate of a solver on held-out tasks: a proposal succeeds if it matches the reference on every test
    (visible and hidden). The search solver chooses its proposal with the first ``n_visible`` tests only."""
    jobs, owner = [], []
    for i, t in enumerate(tasks_held):
        inputs = make_tests(t.func, seed, n_tests)
        jobs.append({"source": t.fix, "func": t.func, "inputs": inputs})
        owner.append((i, "ref", None))
        props = [retrieval_solver(t, train)] if solver == "retrieval" else search_solver(t)
        for p in props:
            jobs.append({"source": p, "func": t.func, "inputs": inputs})
            owner.append((i, "cand", p))
    out = run_batch(jobs, timeout=60.0) or []
    ref, best = {}, {}
    for (i, kind, p), r in zip(owner, out):
        if kind == "ref":
            ref[i] = r["results"]
    for (i, kind, p), r in zip(owner, out):
        if kind != "cand" or i in best or not r.get("ok"):
            continue
        res = r["results"]
        if solver == "search" and res[:n_visible] != ref[i][:n_visible]:
            continue
        best[i] = res == ref[i]
    solved = [best.get(i, False) for i in range(len(tasks_held))]
    return {"held": len(tasks_held), "solved": sum(solved), "rate": sum(solved) / max(1, len(tasks_held))}


# --------------------------------------------------------------------------- the real dataset

_FILE = re.compile(r"^diff --git a/(\S+) b/", re.M)


def patch_files(patch: str) -> set[str]:
    return set(_FILE.findall(patch or ""))


def swesmith_overlap(instance_ids: list[str], patches: list[str], fail_to_pass: list[list[str]], by: str,
                     val_per_mille: int = 150, test_per_mille: int = 150, salt: str = "m16") -> dict:
    """Split SWE-smith instances by ``by`` ("instance" or "repository") with the course hash rule and measure, for
    held-out instances: the share whose patch touches a (repo, file) that a training patch touches, and the share
    with a failing test that is also a failing test of a training instance."""
    from frontierlab.datax.groups import group_split
    repos = [i.split(".", 1)[0] for i in instance_ids]
    keys = instance_ids if by == "instance" else repos
    sp = group_split(keys, val_per_mille, test_per_mille, salt=salt)
    train_files, train_tests = set(), set()
    for r, p, f, s in zip(repos, patches, fail_to_pass, sp):
        if s == "train":
            train_files |= {(r, x) for x in patch_files(p)}
            train_tests |= {(r, x) for x in f}
    held = [(r, p, f) for r, p, f, s in zip(repos, patches, fail_to_pass, sp) if s != "train"]
    file_hit = sum(bool({(r, x) for x in patch_files(p)} & train_files) for r, p, f in held)
    test_hit = sum(bool({(r, x) for x in f} & train_tests) for r, p, f in held)
    both_sides = len({r for r, s in zip(repos, sp) if s != "train"} & {r for r, s in zip(repos, sp) if s == "train"})
    n = max(1, len(held))
    return {"by": by, "held": len(held), "file_overlap": file_hit / n, "test_overlap": test_hit / n,
            "repos_on_both_sides": both_sides}


__all__ = ["REPOS", "FAMILIES", "mutate", "make_tests", "repo_of", "run_batch", "SweTask", "synthesise", "split",
           "jaccard", "retrieval_solver", "search_solver", "solve_and_score", "patch_files", "swesmith_overlap"]
