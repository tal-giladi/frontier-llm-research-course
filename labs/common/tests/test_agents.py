"""Tests for frontierlab.agents (Module 16): sandbox, environment protocol, verifiers and the candidate registry,
the program world and its rewards, multi-turn masks, credit over turns, the agent RL loop, monitors, task
synthesis and group splits."""

import json
import random

import pytest
import torch

from frontierlab.agents import agentrl as R
from frontierlab.agents import codeenv as C
from frontierlab.agents import dsl, monitor, swetasks
from frontierlab.agents import turns as T
from frontierlab.agents.env import Task, determinism_check, reset_check, run_episode
from frontierlab.agents.sandbox import run_python
from frontierlab.model import LM


# --------------------------------------------------------------------------- sandbox

def test_sandbox_runs_and_times_out():
    r = run_python("print(1 + 1)")
    assert r.ok and r.stdout.strip() == "2"
    r = run_python("while True:\n    pass\n", timeout=1.0)
    assert r.timed_out and not r.ok


def test_sandbox_blocks_network_and_subprocess_and_bad_paths():
    r = run_python("import socket\nsocket.create_connection(('example.org', 80))\n")
    assert not r.ok and "blocked by the course sandbox" in r.stderr
    r = run_python("import os\nos.system('echo hi')\n")
    assert not r.ok and "blocked" in r.stderr
    with pytest.raises(ValueError):
        run_python("print(1)", files={"../x.py": "1"})


# --------------------------------------------------------------------------- code environment and verifiers

def test_reset_and_determinism_checks():
    task = C.basic_tasks()[0]
    spec = C.SPEC_BY_NAME[task.data["func"]]

    def disturb(env):
        env.step(("write", {"path": "solution.py", "text": C.solution_file(spec, spec.reference)}))

    ok = reset_check(C.CodeEnv, task, disturb)
    assert ok == {"disturb_changed_state": True, "reset_restores": True, "fresh_matches": True}
    leaky = reset_check(C.LeakyCodeEnv, task, disturb)
    assert leaky["disturb_changed_state"] and not leaky["reset_restores"]
    acts = [("read", {"path": "examples.txt"}), ("write", {"path": "solution.py",
                                                            "text": C.solution_file(spec, spec.reference)})]
    assert determinism_check(lambda: C.CodeEnv("hidden"), task, acts)


def test_episode_with_scripted_agent():
    task = C.basic_tasks()[3]
    spec = C.SPEC_BY_NAME[task.data["func"]]
    script = iter([("ls", {}), ("nope", {}), ("write", {"path": "solution.py", "text": C.solution_file(spec, spec.reference)}),
                   ("run_tests", {}), ("submit", {})])
    traj = run_episode(C.CodeEnv("robust"), task, lambda h: next(script))
    assert "unknown tool" in traj.turns[1].observation
    assert traj.turns[3].observation.startswith("3 of 3")
    assert traj.verdict.passed and not traj.truncated


def test_verifier_report_on_registered_candidates():
    rep = C.verifier_report(C.basic_tasks()[:3])
    s = rep["summary"]
    assert s["visible"]["loophole_accepts"] == s["visible"]["loophole_total"] == 3
    for v in ("hidden", "robust"):
        assert s[v]["loophole_accepts"] == 0 and s[v]["false_rejects"] == 0
    assert s["visible"]["false_rejects"] == 0
    assert len(rep["rows"]) == 3 * len(C.CANDIDATES) * 3


def test_registry_rules():
    with pytest.raises(ValueError):
        C.register("reference", "correct", {})
    with pytest.raises(ValueError):
        C.register("x", "exploit", {})
    task = C.basic_tasks()[0]
    C.register("_test_fixed_stub", "wrong", {"solution.py": task.data["files"]["solution.py"]})
    try:
        rep = C.verifier_report([task], candidates=["_test_fixed_stub"])
        assert all(not r["passed"] for r in rep["rows"])
    finally:
        del C.CANDIDATES["_test_fixed_stub"]


def test_properties_hold_for_reference_outputs():
    rng = random.Random(0)
    for spec in C.SPECS:
        ref = C.reference_fn(spec)
        for _ in range(50):
            a = list(spec.gen(rng))
            assert C.PROPERTIES[spec.name](a, json.loads(json.dumps(ref(*a))))


# --------------------------------------------------------------------------- the program world

def test_programs_and_rewards():
    t = dsl.DslTask(dsl.Func("add", 3), (1, 2, 0))
    assert t.prompt == "1>4;2>5;0>3:"
    assert dsl.run_program("x+3", 7, []) == 10 and dsl.run_program("x*2+1", 4, []) == 9
    tab = dsl.table_program(t)
    assert tab == "T453" and dsl.run_program(tab, 2, [1, 2, 0]) == 5 and dsl.run_program(tab, 9, [1, 2, 0]) == 0
    assert dsl.r_visible(tab, True, t) == 1 and dsl.r_hidden(tab, True, t) == 0 and dsl.r_property(tab, True, t) == 0
    assert dsl.r_gold("x+3", True, t) == 1 and dsl.r_gold("x+3", False, t) == 0
    assert dsl.r_format("x+9", True, t) == 1 and dsl.r_format("x+", True, t) == 0
    assert dsl.r_length("x+3x+3x+3", False, t) == 1.0
    assert dsl.kind_of("T12") == "table" and dsl.kind_of("??") == "other"
    rng = random.Random(0)
    assert dsl.r_randomised("x+3", True, t, rng) == 1 and dsl.r_randomised(tab, True, t, random.Random(1)) == 0


def test_every_rule_is_identified_by_its_visible_pairs():
    for t in dsl.all_tasks():
        good = [f for f in dsl.ALL_FUNCS if dsl.r_visible(f.program, True, t)]
        assert good == [t.func]


def test_splits_are_disjoint():
    for by in ("instance", "function"):
        tr, he = dsl.split_tasks(0, by=by)
        assert not {t.id for t in tr} & {t.id for t in he} and len(tr) + len(he) == 264
    tr, he = dsl.split_tasks(0, by="function")
    assert not {t.func for t in tr} & {t.func for t in he}


# --------------------------------------------------------------------------- multi-turn rollouts

def tiny_policy(seed=0):
    torch.manual_seed(seed)
    return LM(R.dsl_policy_config())


def test_multiturn_masks_and_inserted_observations():
    envs = [T.ProbeTask(f, max_queries=2) for f in dsl.ALL_FUNCS[:4]]
    ro = T.sample_multiturn(tiny_policy(), envs, 16, generator=torch.Generator().manual_seed(0))
    assert torch.all(ro.action_mask * ro.obs_mask == 0)
    for i, ep in enumerate(ro.episodes):
        inserted = [t for t, m in zip(ro.response[i].tolist(), ro.obs_mask[i].tolist()) if m]
        want = sum((dsl.TOKD.encode(o) for o in ep.observations), [])
        assert inserted == want[:len(inserted)] and (len(inserted) == len(want) or ep.truncated)
        assert torch.all(ro.sampler_logp[i][ro.obs_mask[i] > 0] == 0)


def test_scripted_probe_episode():
    env = T.ProbeTask(dsl.Func("mul", 3), max_queries=2, info_bonus=1.0)
    ep = T.Episode()
    for a in ("?2", "?1"):
        ep.actions.append(a)
        ep.observations.append(env.observe(a))
    assert ep.observations == ["=6.", "=3."]
    assert [f.name for f in env.candidates(ep)] == ["mul3"]
    assert env.turn_reward("?1", ep) > 0
    ep.final, ep.finished = "x*3", True
    assert env.final_reward(ep) == 1.0


def test_context_tokens():
    assert T.context_tokens(3, [2, 2, 3], [3, 3, 0]) == [3, 8, 13]
    assert T.context_tokens(3, [2, 2, 3], [3, 3, 0], "window", 1) == [3, 8, 8]


# --------------------------------------------------------------------------- credit over turns

def test_turn_returns_and_advantages():
    g = R.turn_returns([[0.1, 0.2], [0.0]], [1.0, 0.0], gamma=0.5)
    assert g[0] == pytest.approx([0.1 + 0.5 * 0.2 + 0.25 * 1.0, 0.2 + 0.5, 1.0]) and g[1] == [0.0, 0.0]
    adv = R.turn_advantages([[1.0, 1.0], [0.0, 0.0]], group=2)
    assert adv[0][0] > 0 > adv[1][0] and adv[0][0] == pytest.approx(-adv[1][0])
    ti = torch.tensor([[0, 0, 1, 1, -1]])
    assert R.token_advantages([[2.0, -1.0]], ti).tolist() == [[2.0, 2.0, -1.0, -1.0, 0.0]]


def test_outcome_advantages_match_grpo():
    tot = torch.tensor([1.0, 0.0, 0.0, 0.0])
    a = R.outcome_advantages(tot, 4)
    assert a[0] > 0 and torch.allclose(a[1:], a[1].expand(3)) and abs(float(a.sum())) < 1e-5


# --------------------------------------------------------------------------- the loop, monitors, traces

@pytest.fixture(scope="module")
def short_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("m16")
    sft = R.train_sft(root / "sft", "single", steps=30)
    init = str(root / "sft" / "policy.pt")
    base = dict(init=init, steps=4, prompts=4, group=4, eval_every=2, eval_n=12, eval_samples=1, ckpt_every=2,
                traces=6)
    vis = R.AgentRLConfig(run=str(root / "vis"), reward="visible", **base)
    R.train(vis)
    probe_init = str(root / "sftp" / "policy.pt")
    R.train_sft(root / "sftp", "probe", steps=20)
    pr = R.AgentRLConfig(run=str(root / "probe"), task="probe", init=probe_init, max_tokens=16, credit="turn",
                         info_bonus=0.3, loss_on="actions", **{k: v for k, v in base.items() if k != "init"})
    R.train(pr)
    return {"root": root, "vis": vis, "probe": pr, "sft": sft}


def test_loop_logs_and_resumes(short_runs):
    from dataclasses import replace
    rows = [json.loads(x) for x in open(f"{short_runs['vis'].run}/metrics.jsonl")]
    tr = [r for r in rows if r["split"] == "train"]
    assert len(tr) == 4 and all(r["obs_tokens_in_loss"] == 0 for r in tr)
    assert {"gold", "frac_table", "kl_k3", "reward"} <= set(tr[0])
    cont = replace(short_runs["vis"], steps=6)
    R.train(cont)
    rows2 = [json.loads(x) for x in open(f"{cont.run}/metrics.jsonl")]
    assert len([r for r in rows2 if r["split"] == "train"]) == 6


def test_probe_run_and_masking(short_runs):
    rows = [json.loads(x) for x in open(f"{short_runs['probe'].run}/metrics.jsonl")]
    tr = [r for r in rows if r["split"] == "train"]
    assert all(r["obs_tokens_in_loss"] == 0 for r in tr) and "queries" in tr[0]


def test_traces_validate_and_tampering_is_caught(short_runs, tmp_path):
    path = f"{short_runs['vis'].run}/traces.jsonl"
    v = monitor.validate_traces(path)
    assert v["valid"] and v["records"] > 0
    recs = [json.loads(x) for x in open(path)]
    recs[0]["scores"]["gold"] = 1 - recs[0]["scores"]["gold"]
    bad = tmp_path / "t.jsonl"
    bad.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    assert not monitor.validate_traces(bad)["valid"]


def write_log(path, train, evals):
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "metrics.jsonl", "w") as f:
        for i, (r, g) in enumerate(train):
            f.write(json.dumps({"split": "train", "step": i + 1, "reward": r, "gold": g}) + "\n")
        for e in evals:
            f.write(json.dumps({"split": "eval", **e}) + "\n")


def test_audit_flags(tmp_path):
    ev0 = {"frac_rule": 0.85, "frac_table": 0.15, "frac_other": 0.0, "len": 3.5, "trunc": 0.0, "gold_pass": 0.25,
           "visible_pass": 0.35}
    ev1 = {"frac_rule": 0.05, "frac_table": 0.95, "frac_other": 0.0, "len": 4.0, "trunc": 0.0, "gold_pass": 0.0,
           "visible_pass": 0.95}
    write_log(tmp_path / "hack", [(0.4, 0.25)] * 10 + [(1.0, 0.0)] * 10, [ev0, ev1])
    a = monitor.audit_run(tmp_path / "hack")
    assert a["flags"] == ["divergence", "gold_drop", "kind_shift"]
    write_log(tmp_path / "ok", [(0.3, 0.3)] * 10 + [(0.7, 0.7)] * 10, [ev0, {**ev0, "gold_pass": 0.6}])
    assert monitor.audit_run(tmp_path / "ok")["flags"] == []


def test_looks_hardcoded():
    t = C.basic_tasks()[0]
    assert monitor.looks_hardcoded(C.lookup_table_solution(t), t.data["visible"])
    assert not monitor.looks_hardcoded(C.candidate_files("reference", t)["solution.py"], t.data["visible"])


# --------------------------------------------------------------------------- task synthesis and splits

def test_mutations():
    src = "def f(x):\n    return x + 1 if x < 3 else max(x, 2)\n"
    assert any("x - 1" in m for m in swetasks.mutate(src, "swap_binop"))
    assert any("x <= 3" in m for m in swetasks.mutate(src, "flip_compare"))
    assert any("min(x, 2)" in m for m in swetasks.mutate(src, "swap_minmax"))
    assert len(swetasks.mutate(src, "shift_const")) == 6


@pytest.fixture(scope="module")
def synth():
    return swetasks.synthesise()


def test_synthesised_tasks_break_a_test(synth):
    tasks, stats = synth
    assert stats["kept"] == len(tasks) > 40 and all(t.failing for t in tasks)
    assert stats["kept"] + stats["no_failing_test"] + stats["crashed"] == stats["candidates"]


def test_group_splits_and_solvers(synth):
    tasks, _ = synth
    for by in ("function", "repo"):
        tr, he = swetasks.split(tasks, by)
        assert not {t.key[by] for t in tr} & {t.key[by] for t in he}
        assert swetasks.solve_and_score(he, tr, "retrieval")["rate"] == 0.0
    tr, he = swetasks.split(tasks, "instance")
    assert swetasks.solve_and_score(he, tr, "retrieval")["rate"] > 0.5


def test_swesmith_overlap_on_synthetic_ids():
    ids = [f"r{i % 3}__r.abc.func_basic__{i}" for i in range(60)]
    patches = [f"diff --git a/src/m{i % 3}.py b/src/m{i % 3}.py\n" for i in range(60)]
    f2p = [[f"tests/t{i % 3}.py::x"] for i in range(60)]
    inst = swetasks.swesmith_overlap(ids, patches, f2p, "instance")
    repo = swetasks.swesmith_overlap(ids, patches, f2p, "repository")
    assert inst["file_overlap"] == 1.0 and repo["file_overlap"] == 0.0 and repo["repos_on_both_sides"] == 0


def test_hf_agent_smoke(tmp_path):
    from frontierlab.agents import hf_agent as HA
    for task in ("single", "probe"):
        cfg = HA.HFAgentConfig(run=str(tmp_path / task), task=task, smoke=True, steps=1, prompts=2, group=2, eval_n=4,
                               minibatches=1, eval_every=1, credit="turn", split="function" if task == "probe" else "none")
        HA.train(cfg)
        rows = [json.loads(x) for x in open(tmp_path / task / "metrics.jsonl")]
        tr = [r for r in rows if r["split"] == "train"]
        assert tr and tr[0]["obs_tokens_in_loss"] == 0 and tr[0]["sampler_gap"] < 1e-4
