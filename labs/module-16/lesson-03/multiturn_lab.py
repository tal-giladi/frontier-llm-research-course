"""Lab 16.3: multi-turn agentic RL on the probe task: masks, credit over turns, context cost.

    python labs/module-16/lesson-03/multiturn_lab.py                 # free CPU, about 6-10 minutes
    python labs/module-16/lesson-03/multiturn_lab.py --part ab       # only the checks and the cost table (seconds)
    python labs/module-16/lesson-03/multiturn_lab.py --variant main --print

Part A: rollouts of the warm-started probe policy; your ``loss_mask`` and ``spread_over_tokens`` checked against
the loop's tensors on real multi-turn episodes; how many tokens of each episode are observations.

Part B: your ``context_cost`` on the toy episode shape and on a SWE-agent-like shape (PROJECTED token counts),
full history against a window of the last turns.

Part C (the experiment): four arms x 2 seeds x 60 steps from the same warm start, the same tasks and settings:
``outcome`` (one advantage per episode, loss on actions: the baseline), ``all-tokens`` (loss also on inserted
observation tokens: the masking bug), ``outcome+bonus`` (a shaped information bonus of 0.3 x the share of
remaining candidate functions a query removes, still credited per episode) and ``turn+bonus`` (the same bonus
credited per turn by return-to-go). Held-out gold accuracy at the last step, paired by seed against the
baseline, and the behaviour metrics: queries, malformed actions, observation-shaped actions, turn-limit hits.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from frontierlab.agents import agentrl as R
from frontierlab.agents import dsl
from frontierlab.agents.turns import ProbeTask, sample_multiturn
from frontierlab.labkit import load_path
from frontierlab.metrics.jsonl import read_jsonl
from frontierlab.posttrain.arms import t_interval
from frontierlab.posttrain.sft import load_policy

HERE = Path(__file__).resolve().parent
ROOT = Path("runs/m16/l163")
ARMS = {"outcome": {}, "all-tokens": {"loss_on": "all"}, "outcome+bonus": {"info_bonus": 0.3},
        "turn+bonus": {"credit": "turn", "info_bonus": 0.3}}
SEEDS = [0, 1]


def base_cfg(init, steps=60):
    return R.AgentRLConfig(init=str(init), task="probe", max_tokens=16, max_queries=2, steps=steps, eval_every=20,
                           lr=3e-4, prompts=16, group=8)


def part_a(lab, init):
    pol = load_policy(init)
    envs = [ProbeTask(f, 2) for f in dsl.ALL_FUNCS for _ in range(4)]
    ro = sample_multiturn(pol, envs, 16, generator=torch.Generator().manual_seed(0))
    m = lab.loss_mask(ro.action_mask, ro.obs_mask)
    assert torch.equal(m, ro.action_mask), "loss_mask('actions') must equal the sampled-token mask"
    bad = lab.loss_mask(ro.action_mask, ro.obs_mask, "all")
    adv = [float(i % 3) - 1 for i in range(3)]
    for i in range(len(envs)):
        want = R.token_advantages([adv], ro.turn_index[i:i + 1])[0]
        assert torch.equal(lab.spread_over_tokens(adv, ro.turn_index[i]), want)
    a, o = float(ro.action_mask.sum()), float(ro.obs_mask.sum())
    q = sum(len(e.actions) for e in ro.episodes) / len(envs)
    print(f"(A) {len(envs)} warm-start episodes: {q:.2f} queries each; tokens sampled {a:.0f}, inserted "
          f"(observations) {o:.0f} = {o / (a + o):.1%} of the response; the buggy mask adds {float(bad.sum() - m.sum()):.0f} "
          f"tokens to the loss. loss_mask and spread_over_tokens agree with the loop on every episode.")


def part_b(lab):
    print("(B) context cost per episode (tokens): prefill = sum of the context each action starts from")
    shapes = {"toy probe (prompt 2, action 3, observation 3-4)": (2, [3] * 3, [4, 4, 0]),
              "SWE-agent-like, 30 turns (prompt 3,000; action 150; observation 1,200)": (3000, [150] * 30, [1200] * 30),
              "SWE-agent-like, 100 turns": (3000, [150] * 100, [1200] * 100)}
    for name, (p, acts, obs) in shapes.items():
        full = lab.context_cost(p, acts, obs)
        win = lab.context_cost(p, acts, obs, "window", 3)
        print(f"    {name}\n      full history: prefill {full['prefill_tokens']:,}, peak context {full['peak_context']:,}"
              f"   | last 3 turns: prefill {win['prefill_tokens']:,}, peak {win['peak_context']:,}"
              f"   ({full['prefill_tokens'] / max(1, win['prefill_tokens']):.1f}x fewer prefill tokens)")


def summarise(run):
    rows = read_jsonl(Path(run) / "metrics.jsonl")
    ev = [r for r in rows if r["split"] == "eval"]
    tr = [r for r in rows if r["split"] == "train"]
    last = ev[-1]
    return {"gold": last["gold_pass"], "queries": last["queries"], "bad": last["bad_actions"],
            "self_obs": last["self_obs"], "turn_limit": last["turn_limit"], "start": ev[0]["gold_pass"],
            "kl": float(np.mean([r["kl_k3"] for r in tr[-10:]])), "reward": float(np.mean([r["reward"] for r in tr[-10:]]))}


def part_c(init, steps):
    t0 = time.time()
    runs = R.run_arms(base_cfg(init, steps), ARMS, SEEDS, ROOT)
    res = {a: [summarise(r) for r in rs] for a, rs in runs.items()}
    print(f"(C) {len(ARMS)} arms x {len(SEEDS)} seeds x {steps} steps ({time.time() - t0:.0f}s; finished runs are reused)")
    print(f"    start (warm start) held-out gold: {res['outcome'][0]['start']:.3f}")
    print(f"    {'arm':14s} {'gold per seed':>16s} {'vs outcome [95% CI]':>26s} {'queries':>8s} {'bad':>6s} {'self=':>6s} "
          f"{'limit':>6s} {'KL':>6s}")
    for a, rs in res.items():
        g = [r["gold"] for r in rs]
        if a == "outcome":
            cmp = "baseline"
        else:
            m, lo, hi = t_interval([x - y["gold"] for x, y in zip(g, res["outcome"])])
            cmp = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        mean = lambda k: np.mean([r[k] for r in rs])
        print(f"    {a:14s} {' '.join(f'{x:.3f}' for x in g):>16s} {cmp:>26s} {mean('queries'):8.2f} {mean('bad'):6.3f} "
              f"{mean('self_obs'):6.3f} {mean('turn_limit'):6.3f} {mean('kl'):6.3f}")
    (ROOT / "summary.json").write_text(json.dumps(res, indent=2))


MAIN = ("python -m frontierlab.agents.hf_agent --task probe --credit {credit} --loss-on {loss_on} --info-bonus {bonus} "
        "--steps 150 --prompts 32 --group 8 --seed {seed} --run runs/m16/main/l163-{arm}-s{seed}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="abc")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--variant", choices=["cpu", "main", "t4"], default="cpu")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    if a.variant != "cpu":
        extra = "" if a.variant == "main" else " --model Qwen/Qwen3-0.6B-Base --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd --prompts 8"
        for arm, over in ARMS.items():
            for s in SEEDS:
                print(MAIN.format(credit=over.get("credit", "outcome"), loss_on=over.get("loss_on", "actions"),
                                  bonus=over.get("info_bonus", 0.0), seed=s, arm=arm) + extra)
        return
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    init = R.ensure_sft("runs/m16/sft-probe", task="probe", steps=400)
    if "a" in a.part:
        part_a(lab, init)
    if "b" in a.part:
        part_b(lab)
    if "c" in a.part:
        part_c(init, a.steps)


if __name__ == "__main__":
    main()
