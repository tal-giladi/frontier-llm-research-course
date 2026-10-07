"""The Module 16 environment pack: three environments, each with a repository-level split, a reward, and the
settings its RL runs use. ``test_pack.py`` checks any pack with the same layout; ``buggy_pack.py`` is a
colleague's version of this file.

=========  =======================================  ============================  ==========================
name       tasks                                    repository key (split unit)   training reward
=========  =======================================  ============================  ==========================
``code``   6 toy specs x 3 seeds (lesson 16.1)      ``kit-<family>``: 3 kits      ``robust`` verifier
``swe``    SWE-smith-style synthesis (lesson 16.2)  the toy repository: 4         reference tests, sandboxed
``probe``  query-then-answer (lesson 16.3)          the hidden function: 11       gold answer (inputs 0-19)
=========  =======================================  ============================  ==========================

Splits hold out about a third of the groups (1 of 3 kits, 1 of 4 repositories); the probe task holds out 3 of 11
functions with the split its RL loop uses.
"""

from __future__ import annotations

import random

from frontierlab.agents import codeenv as C
from frontierlab.agents import dsl, swetasks
from frontierlab.agents.turns import ProbeTask

HELD_FRAC = 0.34
SPLIT_SEED = 0


def code_tasks():
    return [C.make_task(spec, seed, repo=f"kit-{spec.family}") for seed in range(3) for spec in C.SPECS]


def split_by(items, key, held_frac=HELD_FRAC, seed=SPLIT_SEED) -> dict:
    groups = sorted({key(x) for x in items})
    rng = random.Random(seed)
    rng.shuffle(groups)
    held = set(groups[:max(1, round(held_frac * len(groups)))])
    return {"train": [x for x in items if key(x) not in held], "held": [x for x in items if key(x) in held]}


def swe_reward(task: swetasks.SweTask, source: str) -> float:
    """1 if ``source`` matches the reference on every test of the function (one sandboxed batch)."""
    inputs = swetasks.make_tests(task.func, 0, 12)
    out = swetasks.run_batch([{"source": task.fix, "func": task.func, "inputs": inputs},
                              {"source": source, "func": task.func, "inputs": inputs}])
    if out is None or not out[1].get("ok"):
        return 0.0
    return float(out[0]["results"] == out[1]["results"])


def probe_reward(func: dsl.Func, answer: str, finished: bool = True) -> float:
    from frontierlab.agents.turns import Episode
    ep = Episode(final=answer, finished=finished)
    return ProbeTask(func, 2).final_reward(ep)


def make_pack() -> dict:
    swe, _ = swetasks.synthesise()
    return {
        "code": {"items": code_tasks(), "key": lambda t: t.repo, "env": lambda: C.CodeEnv("robust"),
                 "reward": C.VERIFIERS["robust"]},
        "swe": {"items": swe, "key": lambda t: t.repo, "reward": swe_reward},
        "probe": {"items": list(dsl.ALL_FUNCS), "key": lambda f: f.name, "reward": probe_reward, "split": probe_split,
                  "rl": {"task": "probe", "split": "function", "loss_on": "actions", "credit": "outcome"}},
    }


def probe_split(funcs, key):
    """The probe task uses the function-level split of ``frontierlab.agents.dsl.split_tasks`` (3 of 11 functions
    held out), the one its RL loop and warm start read through ``agentrl.probe_functions``."""
    from frontierlab.agents.agentrl import probe_functions
    tr, he = probe_functions("function", SPLIT_SEED)
    return {"train": [f for f in funcs if f in tr], "held": [f for f in funcs if f in he]}


def splits(pack: dict) -> dict:
    return {name: e.get("split", split_by)(e["items"], e["key"]) for name, e in pack.items()}
