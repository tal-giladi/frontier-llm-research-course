"""Lab 16.5 (extension): evaluation design for computer-use agents, on simulated results.

    python labs/module-16/lesson-05/cua_eval_lab.py          # free CPU, seconds

SIMULATED DATA. Nothing here runs a GUI agent or reproduces an OSWorld number. A benchmark of 369 tasks shaped
like OSWorld's description (30 infeasible tasks, tasks grouped by application) is filled with outcomes from three
made-up agents, so each evaluation decision can be seen changing the reported number:

1. an interval for the success rate: per task (Wilson) and per application (cluster bootstrap);
2. the infeasible-task rule: what an agent that always answers FAIL scores;
3. the step budget: success at 15 and at 50 steps;
4. environment failures: scoring them as agent failures, excluding them, or rerunning them;
5. comparing two agents: paired by task against unpaired.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from frontierlab.labkit import load_path

HERE = Path(__file__).resolve().parent
APPS = {"chrome": 46, "gimp": 26, "libreoffice_calc": 47, "libreoffice_impress": 47, "libreoffice_writer": 23,
        "multi_apps": 101, "os": 24, "thunderbird": 15, "vlc": 17, "vs_code": 23}     # sums to 369 (made up)
N_INFEASIBLE = 30


def benchmark(seed=0):
    rng = np.random.default_rng(seed)
    apps = [a for a, n in APPS.items() for _ in range(n)]
    feasible = np.ones(len(apps), dtype=bool)
    feasible[rng.choice(len(apps), N_INFEASIBLE, replace=False)] = False
    difficulty = {a: rng.uniform(-1.5, 1.0) for a in APPS}
    return apps, feasible, difficulty


def simulate(agent, apps, feasible, difficulty, seed, env_fail=0.05):
    """Per task: final action, whether the state check passed, steps needed (inf if never), environment error."""
    rng = np.random.default_rng(seed)
    out = []
    for a, f in zip(apps, feasible):
        env_err = rng.random() < env_fail
        if not f:
            says_fail = rng.random() < agent["p_detect_infeasible"]
            out.append({"final": "FAIL" if says_fail else "DONE", "ok": False, "steps": np.inf, "env_error": env_err})
            continue
        if agent.get("always_fail"):
            out.append({"final": "FAIL", "ok": False, "steps": np.inf, "env_error": env_err})
            continue
        logit = agent["skill"] + difficulty[a]
        solves = rng.random() < 1 / (1 + np.exp(-logit)) and not env_err
        steps = rng.gamma(2.0, agent["mean_steps"] / 2.0) if solves else np.inf
        out.append({"final": "DONE", "ok": solves, "steps": steps, "env_error": env_err})
    return out


AGENTS = {"agent A": {"skill": -0.4, "mean_steps": 14, "p_detect_infeasible": 0.3},
          "agent B": {"skill": -0.7, "mean_steps": 9, "p_detect_infeasible": 0.6},
          "always FAIL": {"always_fail": True, "p_detect_infeasible": 1.0}}


def main():
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    apps, feasible, diff = benchmark(0)
    res = {name: simulate(ag, apps, feasible, diff, seed=i + 1) for i, (name, ag) in enumerate(AGENTS.items())}
    print("SIMULATED results on an OSWorld-shaped task list (369 tasks, 30 infeasible) — not real agents")
    print("(1)(2) success at 15 steps, with the infeasible rule; 95% intervals per task (Wilson) and per app (cluster bootstrap)")
    scores = {}
    for name, rs in res.items():
        r = np.array([lab.task_reward(f, x["final"], x["ok"] and x["steps"] <= 15) for f, x in zip(feasible, rs)])
        scores[name] = r
        k, n = int(r.sum()), len(r)
        lo, hi = lab.wilson_interval(k, n)
        m, clo, chi = lab.cluster_interval(r.tolist(), apps)
        feas = r[feasible].mean()
        print(f"    {name:12s} {m:.3f}  Wilson [{lo:.3f}, {hi:.3f}]  app-clustered [{clo:.3f}, {chi:.3f}]   "
              f"feasible tasks only {feas:.3f}, infeasible tasks {r[~feasible].mean():.3f}")
    print("(3) step budget: success on feasible tasks at 15 and at 50 steps")
    for name in ("agent A", "agent B"):
        st = [x["steps"] for x, f in zip(res[name], feasible) if f]
        print(f"    {name}: {lab.success_at_budget(st, 15):.3f} at 15 steps, {lab.success_at_budget(st, 50):.3f} at 50 steps")
    print("(4) environment errors (5% of tasks, made up): three scoring policies for agent A at 15 steps")
    rs = res["agent A"]
    err = np.array([x["env_error"] for x in rs])
    r = scores["agent A"]
    rerun = simulate(AGENTS["agent A"], apps, feasible, diff, seed=99, env_fail=0.0)
    r2 = r.copy()
    for i in np.flatnonzero(err):
        x = rerun[i]
        r2[i] = lab.task_reward(feasible[i], x["final"], x["ok"] and x["steps"] <= 15)
    print(f"    counted as failures {r.mean():.3f}; excluded {r[~err].mean():.3f} (n = {int((~err).sum())}); "
          f"rerun once {r2.mean():.3f}")
    print("(5) agent A vs agent B at 15 steps")
    d = scores["agent A"] - scores["agent B"]
    rng = np.random.default_rng(0)
    boots_p = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    boots_u = [scores["agent A"][rng.integers(0, len(d), len(d))].mean() - scores["agent B"][rng.integers(0, len(d), len(d))].mean()
               for _ in range(2000)]
    print(f"    difference {d.mean():+.3f}; paired by task [{np.quantile(boots_p, .025):+.3f}, {np.quantile(boots_p, .975):+.3f}]; "
          f"unpaired [{np.quantile(boots_u, .025):+.3f}, {np.quantile(boots_u, .975):+.3f}]")
    feas_d = (scores["agent A"] - scores["agent B"])[feasible].mean()
    inf_d = (scores["agent A"] - scores["agent B"])[~feasible].mean()
    print(f"    of which feasible tasks {feas_d:+.3f} (per feasible task), infeasible tasks {inf_d:+.3f} (per infeasible task)")


if __name__ == "__main__":
    main()
