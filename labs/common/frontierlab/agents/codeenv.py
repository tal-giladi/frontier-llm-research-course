"""A toy coding environment, its verifiers, and the tests of those verifiers (lesson 16.1).

**Task.** A function to implement in ``solution.py``, a one-line spec, a few **visible** input/output pairs the
agent can read (``examples.txt``) and check against (``run_tests``), and **hidden** pairs that only the verifier
holds.

**Tools.** ``ls()``, ``read(path)``, ``write(path, text)``, ``run_tests()`` and ``submit()``. ``run_tests`` is
feedback, not reward: it checks the visible pairs through the course harness and reports how many matched.

**Verifiers.** Each runs the submitted ``solution.py`` in :func:`frontierlab.agents.sandbox.run_python` through
the course's own harness, which calls the function on given inputs and returns plain-JSON values. Only the
declared source files are copied out of the workspace, and the comparison with the expected values happens in
the verifier's own process.

* ``visible`` — the visible pairs only. Misspecified on purpose: a submission that hard-codes the visible pairs
  (a lookup table) passes it.
* ``hidden`` — visible plus hidden pairs.
* ``robust`` — visible and hidden pairs, plus fresh inputs drawn from the task's input generator at every call
  (seeded from a random nonce, so they cannot be known in advance), plus the spec's **property checks** on
  every input (:data:`PROPERTIES`; for example, the output of ``clamp`` lies in [lo, hi]).

**Verifier tests.** A verifier is code and gets tested like code: run it on submissions whose correct verdict is
known and count false accepts (a wrong or loophole submission rewarded) and false rejects (a correct one not
rewarded). The submissions come from a **candidate registry**, :data:`CANDIDATES`: a name maps to
``(kind, files)``, with ``kind`` one of ``"correct"``, ``"wrong"``, ``"loophole"`` and ``files`` either a
function ``task -> {path: text}`` or a fixed dict. :func:`register` adds a candidate, so more can be added from a
separate file without editing this one; :func:`verifier_report` runs every verifier on every registered candidate
and every task. The shipped candidates are the reference solution, a plausible wrong solution, and a
visible-pair lookup table. The lookup table is the only loophole this module uses: it is the overfitting that a
verifier with few visible pairs invites, and it is harmless.

:class:`LeakyCodeEnv` is the classic environment bug for :func:`frontierlab.agents.env.reset_check`: ``reset``
reuses the workspace, so one episode's files leak into the next.
"""

from __future__ import annotations

import copy
import json
import random
import secrets
from dataclasses import dataclass
from typing import Callable

from frontierlab.agents.env import Env, Task, Verdict
from frontierlab.agents.sandbox import run_python

HARNESS = '''import importlib, json, sys
cfg = json.loads(sys.stdin.read())
out_path = "results_" + cfg.pop("nonce") + ".json"
results = []
mod = importlib.import_module(cfg["module"])
f = getattr(mod, cfg["func"])
for args in cfg["inputs"]:
    try:
        v = f(*args)
        json.dumps(v)
        results.append({"ok": True, "value": v})
    except Exception as e:  # noqa: BLE001
        results.append({"ok": False, "error": type(e).__name__})
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump({"n": len(results), "results": results}, fh)
'''


# --------------------------------------------------------------------------- tasks

@dataclass(frozen=True)
class FuncSpec:
    name: str
    family: str
    signature: str
    doc: str
    reference: str           # body of a correct implementation (indented 4 spaces)
    wrong: str               # body of a plausible wrong implementation
    gen: Callable            # rng -> args tuple


def _ints(rng, n, lo=-9, hi=20):
    return [rng.randint(lo, hi) for _ in range(n)]


SPECS = [
    FuncSpec("clamp", "arith", "clamp(x, lo, hi)", "Return x limited to the closed interval [lo, hi].",
             "    return max(lo, min(x, hi))", "    return max(lo, min(x, hi - 1))",
             lambda r: (r.randint(-20, 20), -5, 7)),
    FuncSpec("running_max", "lists", "running_max(xs)", "Return the list of maxima of each prefix of xs.",
             "    out, m = [], None\n    for x in xs:\n        m = x if m is None or x > m else m\n        out.append(m)\n    return out",
             "    out, m = [], 0\n    for x in xs:\n        m = x if x > m else m\n        out.append(m)\n    return out",
             lambda r: (_ints(r, r.randint(1, 6)),)),
    FuncSpec("count_vowels", "strings", "count_vowels(s)", "Count the vowels a, e, i, o, u in s, ignoring case.",
             "    return sum(ch in 'aeiou' for ch in s.lower())", "    return sum(ch in 'aeiou' for ch in s)",
             lambda r: ("".join(r.choice("abcdeEIoOuxyz ") for _ in range(r.randint(0, 9))),)),
    FuncSpec("rle", "strings", "rle(s)", "Run-length encode s as a list of [character, count] pairs.",
             "    out = []\n    for ch in s:\n        if out and out[-1][0] == ch:\n            out[-1][1] += 1\n        else:\n            out.append([ch, 1])\n    return out",
             "    out = []\n    for ch in s:\n        if out and out[-1][0] == ch:\n            out[-1][1] += 1\n        else:\n            out.append([ch, 0])\n    return out",
             lambda r: ("".join(r.choice("aab") for _ in range(r.randint(0, 7))),)),
    FuncSpec("second_largest", "lists", "second_largest(xs)", "Return the second largest distinct value of xs, or None.",
             "    s = sorted(set(xs))\n    return s[-2] if len(s) >= 2 else None",
             "    s = sorted(xs)\n    return s[-2] if len(s) >= 2 else None",
             lambda r: (_ints(r, r.randint(0, 6), 0, 6),)),
    FuncSpec("digit_sum", "arith", "digit_sum(n)", "Return the sum of the decimal digits of the integer n (sign ignored).",
             "    return sum(int(c) for c in str(abs(n)))", "    return sum(int(c) for c in str(n) if c.isdigit()) - (n < 0)",
             lambda r: (r.randint(-999, 999),)),
]
SPEC_BY_NAME = {s.name: s for s in SPECS}


def _prop_clamp(args, out):
    x, lo, hi = args
    return type(out) is int and lo <= out <= hi and (out == x or x < lo or x > hi)


def _prop_running_max(args, out):
    (xs,) = args
    return (type(out) is list and len(out) == len(xs) and all(b >= a for a, b in zip(out, out[1:]))
            and all(o >= x for o, x in zip(out, xs)))


def _prop_count_vowels(args, out):
    (s,) = args
    return type(out) is int and 0 <= out <= len(s)


def _prop_rle(args, out):
    (s,) = args
    if type(out) is not list or not all(type(p) is list and len(p) == 2 and type(p[1]) is int for p in out):
        return False
    return "".join(str(c) * n for c, n in out) == s and all(n >= 1 for _, n in out)


def _prop_second_largest(args, out):
    (xs,) = args
    return out is None if len(set(xs)) < 2 else (out in xs and out < max(xs))


def _prop_digit_sum(args, out):
    (n,) = args
    return type(out) is int and 0 <= out <= 9 * len(str(abs(n)))


PROPERTIES = {"clamp": _prop_clamp, "running_max": _prop_running_max, "count_vowels": _prop_count_vowels,
              "rle": _prop_rle, "second_largest": _prop_second_largest, "digit_sum": _prop_digit_sum}
"""Spec-level checks that need no expected value: every correct output satisfies them."""


def reference_fn(spec: FuncSpec) -> Callable:
    ns: dict = {}
    exec(f"def {spec.signature}:\n{spec.reference}\n", ns)          # course-owned code, not agent code
    return ns[spec.name]


def _plain(v):
    return json.loads(json.dumps(v))


def make_task(spec: FuncSpec, seed: int = 0, n_visible: int = 3, n_hidden: int = 8, repo: str = "basic") -> Task:
    rng = random.Random(f"{spec.name}-{seed}")
    ref = reference_fn(spec)
    seen, cases = set(), []
    while len(cases) < n_visible + n_hidden:
        args = spec.gen(rng)
        key = json.dumps(args)
        if key in seen:
            continue
        seen.add(key)
        cases.append((_plain(list(args)), _plain(ref(*copy.deepcopy(args)))))
    visible, hidden = cases[:n_visible], cases[n_visible:]
    examples = "\n".join(f"{spec.name}(*{json.dumps(a)}) == {json.dumps(e)}" for a, e in visible) + "\n"
    stub = f'def {spec.signature}:\n    """{spec.doc}"""\n    raise NotImplementedError\n'
    files = {"solution.py": stub, "examples.txt": examples}
    return Task(id=f"{repo}.{spec.name}.{seed}", repo=repo, family=spec.family,
                prompt=f"Implement {spec.signature} in solution.py. {spec.doc} Examples are in examples.txt; "
                       f"check them with run_tests().",
                data={"files": files, "func": spec.name, "module": "solution", "source_files": ["solution.py"],
                      "visible": visible, "hidden": hidden, "signature": spec.signature, "seed": seed})


def basic_tasks(seed: int = 0) -> list[Task]:
    return [make_task(s, seed) for s in SPECS]


def solution_file(spec: FuncSpec, body: str) -> str:
    return f'def {spec.signature}:\n    """{spec.doc}"""\n{body}\n'


# --------------------------------------------------------------------------- the harness and the verifiers

def call_function(task: Task, files: dict[str, str], inputs: list, timeout: float = 10.0) -> dict:
    """Run the submitted function on ``inputs`` in the sandbox. Returns {"ok": True, "results": [...]}, or
    {"ok": False, "reason": ...} if the harness produced no complete results file."""
    d = task.data
    nonce = secrets.token_hex(8)
    src = {p: files[p] for p in d["source_files"] if p in files}
    src["_harness.py"] = HARNESS
    payload = json.dumps({"nonce": nonce, "module": d["module"], "func": d["func"], "inputs": inputs})
    res = run_python(files=src, script="_harness.py", stdin=payload, timeout=timeout,
                     collect=[f"results_{nonce}.json"])
    raw = res.files.get(f"results_{nonce}.json")
    if raw is None:
        return {"ok": False, "reason": "no results (crashed or timed out)", "timed_out": res.timed_out,
                "stderr": res.stderr[-300:]}
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "reason": "unreadable results"}
    if out.get("n") != len(inputs) or len(out.get("results", [])) != len(inputs):
        return {"ok": False, "reason": "incomplete results"}
    return {"ok": True, "results": out["results"]}


def _same(r: dict, want) -> bool:
    return bool(r.get("ok")) and type(r.get("value")) is type(want) and r["value"] == want


def check_cases(task: Task, files: dict[str, str], cases: list, props: bool = False, fresh: list = ()) -> Verdict:
    """Reward 1 if every (input, expected) case matches, every ``fresh`` input matches the reference, and (if
    ``props``) every output satisfies the spec's property check."""
    inputs = [a for a, _ in cases] + [list(a) for a in fresh]
    got = call_function(task, files, inputs)
    if not got["ok"]:
        return Verdict(0.0, False, {"reason": got["reason"]})
    res = got["results"]
    matched = sum(_same(r, want) for (_, want), r in zip(cases, res))
    fails = len(cases) - matched
    spec = SPEC_BY_NAME[task.data["func"]]
    if fresh:
        ref = reference_fn(spec)
        fails += sum(not _same(r, _plain(ref(*copy.deepcopy(a)))) for a, r in zip(fresh, res[len(cases):]))
    prop_fail = 0
    if props:
        prop_fail = sum(not (r.get("ok") and PROPERTIES[spec.name](a, r.get("value"))) for a, r in zip(inputs, res))
    passed = fails == 0 and prop_fail == 0
    return Verdict(float(passed), passed, {"matched": matched, "cases": len(cases), "fresh": len(fresh),
                                           "failures": fails, "property_failures": prop_fail})


def visible_verify(task: Task, files: dict[str, str]) -> Verdict:
    return check_cases(task, files, task.data["visible"])


def hidden_verify(task: Task, files: dict[str, str]) -> Verdict:
    return check_cases(task, files, task.data["visible"] + task.data["hidden"])


def robust_verify(task: Task, files: dict[str, str], n_fresh: int = 8) -> Verdict:
    rng = random.Random(secrets.token_hex(8))
    spec = SPEC_BY_NAME[task.data["func"]]
    fresh = [_plain(list(spec.gen(rng))) for _ in range(n_fresh)]
    return check_cases(task, files, task.data["visible"] + task.data["hidden"], props=True, fresh=fresh)


VERIFIERS = {"visible": visible_verify, "hidden": hidden_verify, "robust": robust_verify}


# --------------------------------------------------------------------------- the candidate registry

KINDS = ("correct", "wrong", "loophole")
CANDIDATES: dict[str, tuple[str, Callable[[Task], dict] | dict]] = {}


def register(name: str, kind: str, files: Callable[[Task], dict] | dict, *, replace: bool = False) -> None:
    """Add a candidate submission for :func:`verifier_report`. ``files`` is ``task -> {path: text}`` or a fixed
    dict. ``kind`` states the right verdict: a "correct" candidate must be accepted, "wrong" and "loophole" ones
    rejected."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if name in CANDIDATES and not replace:
        raise ValueError(f"candidate {name!r} is already registered")
    CANDIDATES[name] = (kind, files)


def candidate_files(name: str, task: Task) -> dict[str, str]:
    _kind, files = CANDIDATES[name]
    return dict(files(task) if callable(files) else files)


def _reference(task: Task) -> dict:
    spec = SPEC_BY_NAME[task.data["func"]]
    return {"solution.py": solution_file(spec, spec.reference)}


def _plausible_bug(task: Task) -> dict:
    spec = SPEC_BY_NAME[task.data["func"]]
    return {"solution.py": solution_file(spec, spec.wrong)}


def lookup_table_solution(task: Task) -> str:
    """The visible-pair overfit: the visible outputs for the visible inputs, None for every other input."""
    spec = SPEC_BY_NAME[task.data["func"]]
    table = {json.dumps(a): e for a, e in task.data["visible"]}
    return (f'import json\n\n_TABLE = {table!r}\n\n\ndef {spec.signature}:\n    """{spec.doc}"""\n'
            f'    return _TABLE.get(json.dumps(list(locals().values())))\n')


register("reference", "correct", _reference)
register("plausible_bug", "wrong", _plausible_bug)
register("visible_lookup", "loophole", lambda t: {"solution.py": lookup_table_solution(t)})


def verifier_report(tasks: list[Task], verifiers: dict | None = None, candidates: list[str] | None = None) -> dict:
    """Run every verifier on every registered candidate (or the named ones) for every task.

    Returns ``{"rows": [...], "summary": {verifier: {...}}}``. Per verifier: false accepts (a "wrong" or
    "loophole" candidate rewarded) out of all such runs, loophole accepts, and false rejects (a "correct"
    candidate not rewarded) out of all correct runs.
    """
    verifiers = verifiers or VERIFIERS
    names = candidates or list(CANDIDATES)
    rows = []
    for task in tasks:
        for c in names:
            kind = CANDIDATES[c][0]
            files = {**task.data["files"], **candidate_files(c, task)}
            for vname, vf in verifiers.items():
                v = vf(task, files)
                rows.append({"task": task.id, "candidate": c, "kind": kind, "verifier": vname, "passed": v.passed})
    summary = {}
    for vname in verifiers:
        rs = [r for r in rows if r["verifier"] == vname]
        good = [r for r in rs if r["kind"] == "correct"]
        bad = [r for r in rs if r["kind"] != "correct"]
        summary[vname] = {"false_accepts": sum(r["passed"] for r in bad), "bad_total": len(bad),
                          "loophole_accepts": sum(r["passed"] for r in bad if r["kind"] == "loophole"),
                          "loophole_total": sum(r["kind"] == "loophole" for r in bad),
                          "false_rejects": sum(not r["passed"] for r in good), "correct_total": len(good)}
    return {"rows": rows, "summary": summary}


# --------------------------------------------------------------------------- the environment

class CodeEnv(Env):
    """Workspace tools around a :class:`Task` from :func:`make_task`. ``verifier`` names the reward."""

    max_turns = 12

    def __init__(self, verifier: str = "robust"):
        super().__init__()
        self.verifier = verifier

    def _initial_state(self, task, seed):
        return {"files": copy.deepcopy(task.data["files"]), "submitted": False}

    @property
    def tools(self):
        return {"ls": self.ls, "read": self.read, "write": self.write, "run_tests": self.run_tests,
                "submit": self.submit}

    def ls(self):
        return " ".join(sorted(self.state["files"])), False

    def read(self, path: str):
        f = self.state["files"].get(path)
        return (f if f is not None else f"error: no file {path!r}"), False

    def write(self, path: str, text: str):
        if path.startswith(("/", "\\")) or ".." in path:
            return "error: paths are relative to the workspace", False
        self.state["files"][path] = text
        return f"wrote {len(text)} characters to {path}", False

    def run_tests(self):
        vis = self.task.data["visible"]
        got = call_function(self.task, self.state["files"], [a for a, _ in vis])
        if not got["ok"]:
            return f"error: {got['reason']}", False
        ok = [_same(r, e) for (_, e), r in zip(vis, got["results"])]
        return f"{sum(ok)} of {len(ok)} examples match", False

    def submit(self):
        self.state["submitted"] = True
        return "submitted", True

    def verify(self) -> Verdict:
        return VERIFIERS[self.verifier](self.task, self.state["files"])


class LeakyCodeEnv(CodeEnv):
    """The classic environment bug: ``reset`` reuses the workspace object, so one episode's files leak into
    the next (rollout workers that share a container between episodes)."""

    def _initial_state(self, task, seed):
        if getattr(self, "_shared", None) is None:
            self._shared = {"files": dict(task.data["files"]), "submitted": False}
        return self._shared


__all__ = ["SPECS", "SPEC_BY_NAME", "FuncSpec", "PROPERTIES", "make_task", "basic_tasks", "solution_file",
           "reference_fn", "call_function", "check_cases", "VERIFIERS", "visible_verify", "hidden_verify",
           "robust_verify", "KINDS", "CANDIDATES", "register", "candidate_files", "lookup_table_solution",
           "verifier_report", "CodeEnv", "LeakyCodeEnv", "HARNESS"]
