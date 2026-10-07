"""A colleague's environment pack. Their message:

    "Pack is ready for the agent RL run. Same three environments as the course pack, but I made a few
    simplifications so it runs faster: workers reuse their environment between episodes, the code reward is
    the quick verifier (the hidden-test one was slow), the split is by task id so both sides get every kit,
    and the probe loss covers the whole response so the policy also learns to read tool output. A first run
    looked great: training reward 0.95 on code within 40 steps."

There are four bugs. Find each one with a check in ``test_pack.py`` and name the training-time symptom it would
cause. Do not fix them here; write the fixes in your report.
"""

from __future__ import annotations

from frontierlab.agents import codeenv as C
from frontierlab.agents import dsl, swetasks

import pack as course   # noqa: E402  (the course pack, for the parts that are unchanged)


def make_pack() -> dict:
    swe, _ = swetasks.synthesise()
    return {
        "code": {"items": course.code_tasks(), "key": lambda t: t.id, "env": lambda: C.LeakyCodeEnv("visible"),
                 "reward": C.VERIFIERS["visible"]},
        "swe": {"items": swe, "key": lambda t: t.repo, "reward": course.swe_reward},
        "probe": {"items": list(dsl.ALL_FUNCS), "key": lambda f: f.name, "reward": course.probe_reward,
                  "split": course.probe_split,
                  "rl": {"task": "probe", "split": "function", "loss_on": "all", "credit": "outcome"}},
    }


def splits(pack: dict) -> dict:
    return course.splits(pack)
