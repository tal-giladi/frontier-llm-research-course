"""Checks for an environment pack (the project's correctness suite).

    pytest labs/module-16/project                    # the course pack: every test passes
    PACK=buggy pytest labs/module-16/project         # the colleague's pack: which tests fail, and why?

Each test is one property an RL run silently depends on: groups never cross the split, reset restores the
initial state, the training reward rejects the registered wrong and loophole candidates and accepts the correct
ones, and the loss never covers inserted observation tokens.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from frontierlab.agents import agentrl as R  # noqa: E402
from frontierlab.agents import codeenv as C  # noqa: E402
from frontierlab.agents import dsl  # noqa: E402
from frontierlab.agents.env import reset_check  # noqa: E402
from frontierlab.agents.turns import ProbeTask, sample_multiturn  # noqa: E402
from frontierlab.labkit import load_path  # noqa: E402

MOD = load_path(str(HERE / ("buggy_pack.py" if os.environ.get("PACK") == "buggy" else "pack.py")))


@pytest.fixture(scope="module")
def pack():
    return MOD.make_pack()


@pytest.fixture(scope="module")
def splits(pack):
    return MOD.splits(pack)


@pytest.mark.parametrize("name", ["code", "swe", "probe"])
def test_no_group_crosses_the_split(pack, splits, name):
    sp = splits[name]
    repo_of = {"code": lambda t: t.repo, "swe": lambda t: t.repo, "probe": lambda f: f.name}[name]
    assert sp["held"] and sp["train"]
    assert not {repo_of(x) for x in sp["train"]} & {repo_of(x) for x in sp["held"]}, \
        f"{name}: a repository appears on both sides of the split"


def test_code_env_resets(pack):
    task = pack["code"]["items"][0]
    spec = C.SPEC_BY_NAME[task.data["func"]]
    chk = reset_check(pack["code"]["env"], task, lambda env: env.step(
        ("write", {"path": "solution.py", "text": C.solution_file(spec, spec.reference)})))
    assert chk["disturb_changed_state"] and chk["reset_restores"] and chk["fresh_matches"]


def test_code_reward_rejects_registered_loopholes(pack):
    tasks = pack["code"]["items"][:6]
    rep = C.verifier_report(tasks, verifiers={"pack": pack["code"]["reward"]})
    s = rep["summary"]["pack"]
    assert s["false_rejects"] == 0, "the reward rejects a correct solution"
    assert s["false_accepts"] == 0, f"the reward accepts {s['false_accepts']} wrong or loophole submissions"


def test_swe_reward(pack):
    t = pack["swe"]["items"][0]
    assert pack["swe"]["reward"](t, t.fix) == 1.0
    assert pack["swe"]["reward"](t, t.buggy) == 0.0


def test_probe_reward(pack):
    reward = pack["probe"]["reward"]
    f = dsl.Func("add", 3)
    assert reward(f, "x+3") == 1.0 and reward(f, "x*2") == 0.0 and reward(f, "T456") == 0.0
    assert reward(f, "x+3", finished=False) == 0.0


def test_probe_loss_excludes_observations(pack):
    rl = pack["probe"]["rl"]
    torch.manual_seed(0)
    from frontierlab.model import LM
    pol = LM(R.dsl_policy_config())
    ro = sample_multiturn(pol, [ProbeTask(f, 2) for f in dsl.ALL_FUNCS], 16, generator=torch.Generator().manual_seed(1))
    mask = ro.action_mask if rl["loss_on"] == "actions" else (ro.action_mask + ro.obs_mask).clamp(max=1.0)
    assert float((mask * ro.obs_mask).sum()) == 0.0, "observation tokens are in the loss"
