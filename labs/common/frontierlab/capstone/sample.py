"""A sample capstone package for review practice (lesson 20.2): GSPO vs GRPO stability, by a fictional learner.

**Constructed fixture.** Every number and run card here is invented for the exercise: no run produced them, and they
say nothing about GSPO or GRPO. They are shaped like the outputs of ``labs/module-14/lesson-02/compare_lab.py`` so
that the review is realistic.

* :func:`write_sample` ``(path, flawed=True)`` writes the package as submitted, with ten planted weaknesses
  (:data:`PLANTED`); ``flawed=False`` writes the package after a correct revision, which the checker passes.
* :func:`apply_reruns` simulates the new runs a reviewer's questions require (the learner cannot rerun a fixture):
  it writes the rerun outputs into a flawed package and leaves every other fix to the learner.
* :func:`reference_log` is a revision log that fixes everything, for the reference solution.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from frontierlab.capstone.package import run_name
from frontierlab.capstone.review import revision_kind

SEEDS = [0, 1, 2]
PER_SEED = {   # final held-out accuracy (higher is better) and entropy drop (lower is better), constructed
    "grpo": {"acc": [0.402, 0.371, 0.388], "entropy_drop": [0.19, 0.18, 0.20]},
    "gspo": {"acc": [0.381, 0.366, 0.392], "entropy_drop": [0.15, 0.17, 0.16]},
    "gspo-stale4": {"acc": [0.352, 0.341, 0.367], "entropy_drop": [0.17, 0.18, 0.18]},
    "control-random": {"acc": [0.215, 0.230, 0.221], "entropy_drop": [0.05, 0.06, 0.05]},
}
LR = {"grpo": 3e-4, "gspo": 1e-3, "gspo-stale4": 1e-3, "control-random": 3e-4}
ARMS = {"grpo": "baseline", "gspo": "reproduction", "gspo-stale4": "extension", "control-random": "control"}
COMPARISONS = [
    {"name": "reproduction", "a": "gspo", "b": "grpo", "metric": "acc", "changed": ["args.objective", "args.lr"]},
    {"name": "reproduction-stability", "a": "gspo", "b": "grpo", "metric": "entropy_drop",
     "changed": ["args.objective", "args.lr"]},
    {"name": "extension", "a": "gspo-stale4", "b": "gspo", "metric": "acc", "changed": ["args.staleness"]},
    {"name": "control", "a": "grpo", "b": "control-random", "metric": "acc", "changed": ["args.reward", "args.lr"]},
]
INTERVALS = {   # constructed 95% intervals (paired hierarchical bootstrap over 3 seeds x 300 held-out prompts)
    "reproduction": (-0.031, 0.017), "reproduction-stability": (-0.052, -0.009),
    "extension": (-0.041, -0.012), "control": (0.141, 0.189),
}
PLANTED = ("CONTRACT_STATUS", "CONTRACT_UNCHECKED", "RUNCARD_PARENT", "TUNING", "PARITY", "SEEDS", "UNC_SEEDS",
           "UNC_DECISION", "CLAIM_DIRECTION", "CLAIM_SCOPE", "REPORT_OVERCLAIM")


def _card(arm: str, seed: int, steps: int = 120, parent: str | None = "auto") -> dict:
    if parent == "auto":
        parent = ("m12-sft" if arm == "grpo" else run_name("gspo", seed) if arm == "gspo-stale4"
                  else run_name("grpo", seed))
    return {"run": run_name(arm, seed), "question": f"capstone gspo: arm {arm}", "parent_run": parent,
            "git": {"commit": "5e1f00d", "dirty": False},
            "hardware": {"python": "3.12.13", "torch": "2.14.1+cpu", "platform": "Windows-11", "cpu_threads": 16},
            "config": {"policy": "Module 12 toy policy", "hidden_size": 128, "num_hidden_layers": 4, "vocab_size": 64},
            "args": {"init": "runs/m12/sft/policy.pt", "objective": "gspo" if arm.startswith("gspo") else "grpo",
                     "reward": "random" if arm == "control-random" else "strict", "lr": LR[arm], "steps": steps,
                     "seed": seed, "prompts": 16, "group": 8, "epochs": 1, "minibatches": 4, "kl_beta": 0.0,
                     "staleness": 4 if arm == "gspo-stale4" else 0, "temperature": 1.0, "eval_n": 300},
            "data": {"name": "Module 12 toy tasks", "revision": "tasks-v1"},
            "data_files": {}, "budget": {"steps": steps, "prompts_per_step": 16, "responses": steps * 16 * 8}}


def _comparison(c: dict, seeds_a: list[int], seeds_b: list[int], decision: str | None = None,
                ci: tuple | None = None) -> dict:
    from frontierlab.capstone.uncertainty import decide
    a = [PER_SEED[c["a"]][c["metric"]][s] for s in seeds_a]
    b = [PER_SEED[c["b"]][c["metric"]][s] for s in seeds_b]
    m = float(np.mean(a) - np.mean(b))
    lo, hi = ci or INTERVALS[c["name"]]
    lo, hi = min(lo, m), max(hi, m)
    return {"name": c["name"], "a": c["a"], "b": c["b"], "metric": c["metric"], "n_seeds": min(len(a), len(b)),
            "per_seed": {"a": a, "b": b}, "mean_diff": round(m, 6), "ci": [lo, hi],
            "method": "paired hierarchical bootstrap over seeds and 300 held-out prompts (4000 resamples)",
            "decision": decision or decide(m, lo, hi, 0.02)}


CONTRACT = """# Experiment contract — capstone: GSPO vs GRPO stability

## Question and decision

- **Question:** at equal rollout budget from the Module 12 SFT start, does GSPO train more stably than GRPO
  (entropy drop, gradient spikes, held-out accuracy drawdown) without losing held-out accuracy by more than 0.02?
- **Decision it informs:** which objective the learner's Module 16 agent runs use.

## Hypothesis

- **Hypothesis:** GSPO's entropy drop is lower than GRPO's; held-out accuracy is equivalent within 0.02.
{status}

## Baseline

- **Baseline run card:** `runs/grpo-s<seed>`, parent `m12-sft`.
- **How it was tuned:** learning rates 1e-4, 3e-4, 1e-3 on training pass rate, tuning seed 100, 60 steps per trial.

## What changes and what is held fixed

- **Changed variable:** the objective with its paired settings and its tuned learning rate.
- **Held fixed:** the Module 12 SFT start, 16 prompts x 8 responses, 120 steps, temperature 1, KL off, seeds 0-2,
  Eval v2 pins, software versions.

## Comparison axis

Equal tokens (equal rollouts: same prompts, group size and steps). It does not answer which objective is cheaper per
step or what happens at another rollout budget.

## Budget

- Free CPU, about 25 minutes for 12 runs and 6 tuning runs; main path PROJECTED 27-44 GPU-hours.
- No teacher, judge or generator compute.
- Tuning budget per arm: three learning rates.

## Metrics and decision rule

- **Primary metric:** final held-out accuracy (300 prompts), paired hierarchical bootstrap over seeds and prompts, 95%.
- **Secondary metrics:** entropy drop, gradient-norm spikes, clip fraction.
- **Noise floor:** GRPO's seed standard deviation and the minimum detectable effect for 3 seeds.
- **Decision rule:** equivalent if the 95% interval lies inside [-0.02, +0.02]; a higher or lower if it excludes zero
  otherwise; inconclusive if not.

## Correctness checks (must pass before results count)

- [x] gradient weights of both objectives match lesson 14.1's reference
- [x] same prompts, eval items and seeds for every arm
- [{parity}] budget parity verified from the run cards

## Fallback evidence

Published curves (GSPO Figure 1, ScaleRL Figure 5a), labelled analysis of published results.

## Limits of the conclusion

A dense toy policy, 2-4-token answers, 120 steps; nothing about MoE or Routing Replay; one tuning seed.

## Changes after results

{changes}
"""

REPORT = """# Capstone report — GSPO vs GRPO

## Result

GSPO's entropy drop was lower than GRPO's: -0.030 [-0.052, -0.009]. {acc_sentence}

The extension (GSPO with staleness bound 4) lost {ext:+.3f} accuracy against on-policy GSPO.

## Limits

Dense toy policy, short answers, 120 steps, three seeds.
"""


def write_sample(path, flawed: bool = True) -> Path:
    """Write the sample package (constructed numbers). ``flawed=False``: the correctly revised version."""
    pkg = Path(path)
    if pkg.exists():
        shutil.rmtree(pkg)
    (pkg / "runs").mkdir(parents=True)
    ext_seeds = [0] if flawed else SEEDS
    claim = {"claim": "gspo", "axis": "tokens", "metric": "final held-out accuracy (300 prompts)",
             "lower_is_better": False, "margin": 0.02, "seeds": SEEDS, "external_parents": ["m12-sft"],
             "tuning": {"grpo": 1 if flawed else 3, "gspo": 3, "gspo-stale4": 3, "control-random": 1 if flawed else 3},
             "arms": {a: {"role": r} for a, r in ARMS.items()},
             "comparisons": [{k: c[k] for k in ("name", "a", "b", "changed")} for c in COMPARISONS]}
    if flawed:
        claim["arms"]["gspo-stale4"]["seeds"] = ext_seeds
    (pkg / "claim.yaml").write_text(yaml.safe_dump(claim, sort_keys=False))
    for arm in ARMS:
        for s in (ext_seeds if arm == "gspo-stale4" else SEEDS):
            card = _card(arm, s, steps=160 if (flawed and arm == "gspo-stale4") else 120,
                         parent=None if (flawed and arm == "control-random" and s == 1) else "auto")
            d = pkg / "runs" / run_name(arm, s)
            d.mkdir(parents=True)
            (d / "run_card.yaml").write_text(yaml.safe_dump(card, sort_keys=False))
    comps = []
    for c in COMPARISONS:
        if c["name"] == "extension" and flawed:
            comps.append(_comparison(c, [0], [0], ci=(-0.061, 0.004)))
        elif c["name"] == "reproduction" and flawed:
            comps.append(_comparison(c, SEEDS, SEEDS, decision="a_higher"))
        else:
            comps.append(_comparison(c, SEEDS, SEEDS))
    g = np.array(PER_SEED["grpo"]["acc"])
    sd = float(g.std(ddof=1))
    results = {"claim": "gspo", "constructed_fixture": True, "comparisons": comps,
               "noise_floor": {"arm": "grpo", "seed_std": sd, "n_seeds": 3, "mde": 2.8 * sd * (2 / 3) ** 0.5}}
    (pkg / "results.json").write_text(json.dumps(results, indent=1))
    claims = [
        {"id": "stability", "label": "MEASURED", "scope": "course", "comparison": "reproduction-stability",
         "direction": "lower", "text": "GSPO's entropy drop was lower than GRPO's at course scale."},
        {"id": "accuracy", "label": "MEASURED", "scope": "course", "comparison": "reproduction",
         "direction": "higher" if flawed else "no-difference-detected",
         "text": "GSPO reached higher held-out accuracy than GRPO." if flawed else
                 "No accuracy difference between GSPO and GRPO was detected (interval includes zero; MDE 0.036)."},
        {"id": "staleness", "label": "MEASURED", "scope": "course", "comparison": "extension",
         "direction": "no-difference-detected" if flawed else "lower",
         "text": "Staleness 4 did not change GSPO's accuracy." if flawed else
                 "With staleness bound 4, GSPO's held-out accuracy was lower than on-policy GSPO's at course scale."},
        {"id": "source", "label": "PUBLICLY DOCUMENTED", "scope": "frontier",
         "source": "GSPO (arXiv 2507.18071) sections 5.1 and 5.3",
         "text": "The GSPO paper reports that GSPO removes the need for Routing Replay in MoE RL."},
    ]
    if flawed:
        claims.append({"id": "moe", "label": "MEASURED", "scope": "frontier", "comparison": "reproduction-stability",
                       "direction": "lower",
                       "text": "Our results show GSPO will remove the need for Routing Replay in MoE training."})
    (pkg / "claims.yaml").write_text(yaml.safe_dump(claims, sort_keys=False, width=1000))
    status = "" if flawed else ("- **Status:** reported effect for stability (GSPO sections 5.1-5.3, on MoE); may not "
                                "appear at this scale for a dense toy policy.")
    changes = ("None." if flawed else
               "After review: GRPO re-tuned with the same three learning rates as GSPO (3e-4 selected again, results "
               "unchanged); the extension rerun with seeds 0-2 at 120 steps, which changed its result from "
               "'inconclusive' (one seed, 160 steps) to 'a lower'; the accuracy claim reworded to follow the rule.")
    (pkg / "contract.md").write_text(CONTRACT.format(status=status, parity=" " if flawed else "x", changes=changes),
                                     encoding="utf-8")
    ext = next(c for c in comps if c["name"] == "extension")["mean_diff"]
    acc = ("This proves GSPO is the better objective: it also reached higher held-out accuracy (+0.007 points)."
           if flawed else
           "Held-out accuracy: -0.007 [-0.031, +0.017], inconclusive; no difference detected (MDE 0.036).")
    (pkg / "report.md").write_text(REPORT.format(acc_sentence=acc, ext=ext), encoding="utf-8")
    return pkg


def apply_reruns(path) -> list[str]:
    """Write the outputs of the reruns the review asks for into a flawed package; returns the new run folders."""
    pkg = Path(path)
    tmp = pkg.parent / (pkg.name + "-revised-tmp")
    write_sample(tmp, flawed=False)
    new = []
    try:
        for s in SEEDS:
            n = run_name("gspo-stale4", s)
            dst = pkg / "runs" / n
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(tmp / "runs" / n, dst)
            new.append(f"runs/{n}")
        claim = yaml.safe_load((pkg / "claim.yaml").read_text())
        claim["arms"]["gspo-stale4"].pop("seeds", None)
        claim["tuning"]["grpo"] = claim["tuning"]["control-random"] = 3
        (pkg / "claim.yaml").write_text(yaml.safe_dump(claim, sort_keys=False))
        res = json.loads((pkg / "results.json").read_text())
        fixed = {c["name"]: c for c in json.loads((tmp / "results.json").read_text())["comparisons"]}
        res["comparisons"] = [fixed["extension"] if c["name"] == "extension" else c for c in res["comparisons"]]
        (pkg / "results.json").write_text(json.dumps(res, indent=1))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return new


def reference_log(before) -> list[dict]:
    """A revision log that fixes every problem in ``before`` (the reference solution's log)."""
    text = {
        "CONTRACT_STATUS": "added the hypothesis status (reported effect on MoE; may not appear at this scale)",
        "CONTRACT_UNCHECKED": "ran the package checker; ticked the parity check only after the reruns passed it",
        "RUNCARD_PARENT": "set control-random-s1's parent_run to grpo-s1, the run it branches from",
        "TUNING": "re-tuned GRPO with the same three learning rates as GSPO; 3e-4 selected again",
        "PARITY": "reran the extension at 120 steps, the declared budget",
        "SEEDS": "reran the extension with seeds 0, 1 and 2",
        "UNC_SEEDS": "reran the extension with seeds 0, 1 and 2",
        "UNC_DECISION": "replaced the reported decision with the one the stated rule gives: inconclusive",
        "CLAIM_DIRECTION": "reworded the accuracy claim to 'no difference detected', with the interval and the MDE",
        "CLAIM_SCOPE": "deleted the MoE claim; the paper's Routing Replay result is quoted as PUBLICLY DOCUMENTED only",
        "REPORT_OVERCLAIM": "removed 'proves' and the sentence it was in",
    }
    reruns = [f"runs/{run_name('gspo-stale4', s)}" for s in SEEDS]
    out = []
    for p in before:
        kind = revision_kind(p.code)
        e = {"problem": p.key, "kind": kind, "change": text.get(p.code, "fixed"), "result_changed": False}
        if kind == "rerun":
            e["files"] = reruns if p.code != "TUNING" else ["claim.yaml"]
            e["result_changed"] = p.code in ("PARITY", "SEEDS", "UNC_SEEDS")
        out.append(e)
    return out


def revise_by_hand(path) -> None:
    """The edits a learner makes by hand after :func:`apply_reruns` (the reference solution)."""
    pkg = Path(path)
    clean = Path(str(pkg) + "-clean-tmp")
    write_sample(clean, flawed=False)
    try:
        for f in ("contract.md", "claims.yaml", "report.md", "results.json"):
            shutil.copyfile(clean / f, pkg / f)
        d = pkg / "runs" / run_name("control-random", 1)
        shutil.copyfile(clean / "runs" / run_name("control-random", 1) / "run_card.yaml", d / "run_card.yaml")
    finally:
        shutil.rmtree(clean, ignore_errors=True)


DEFENCE = {   # reference answers, keyed by question id (G, C) or by problem code (P questions)
    "G1": "Equal tokens: every arm sees the same prompts, group size and steps, because the claim is about stability "
          "per unit of training. It does not answer which objective is cheaper per step; contract.md says so.",
    "G2": "After review, GRPO had the same three learning-rate trials as GSPO (claim.yaml, tuning). Its selected rate "
          "did not change, so the comparison numbers in results.json did not change either.",
    "G3": "GRPO's seed std is 0.0155, so the minimum detectable effect with three seeds is 0.036 (results.json, "
          "noise_floor). The accuracy difference of -0.007 [-0.031, +0.017] is far below it: no difference detected.",
    "G4": "With 10x the budget we would run lesson 14.2's main path on Qwen3-1.7B-Base with five seeds "
          "(PROJECTED 27-44 GPU-hours, contract.md). Two inconclusive stability results at that scale would make us drop "
          "GSPO for dense models.",
    "C1": "GSPO's clip fraction is much higher by design, so clip fraction itself cannot be a stability metric here. "
          "The entropy drop is not computed from clipping; its interval is [-0.052, -0.009] (results.json).",
    "C2": "A dense toy policy tests whether the sequence ratio changes stability without experts. The Routing Replay "
          "part needs an MoE model and is quoted only as PUBLICLY DOCUMENTED in claims.yaml.",
    "C3": "The random-reward control reached 0.222 against GRPO's 0.387: grpo minus control is +0.165 [+0.141, +0.189] "
          "(results.json, control). The verifier's signal did the work.",
    "CONTRACT_STATUS": "The status was missing; contract.md now gives 'reported effect' for the MoE source and 'may not "
                       "appear at this scale' for our dense setting. This was decided before the runs and only written later.",
    "CONTRACT_UNCHECKED": "The parity box was unticked because the checker had not been run. It is ticked now, after "
                          "the reruns passed it (contract.md, correctness checks).",
    "RUNCARD_PARENT": "control-random-s1 branches from grpo-s1 like the other control seeds. Its card in "
                      "runs/control-random-s1 now says so.",
    "TUNING": "GRPO got one learning-rate trial against GSPO's three. We re-tuned it with the same three; it chose "
              "3e-4 again (claim.yaml), so results.json is unchanged.",
    "PARITY": "The extension ran 160 steps instead of 120, so it was not on equal tokens. We reran it at 120 steps; "
              "the new runs are runs/gspo-stale4-s0 to s2.",
    "SEEDS": "The extension had seed 0 only. The rerun used seeds 0, 1 and 2, the same as GSPO (runs/gspo-stale4-s1).",
    "UNC_SEEDS": "With one seed there was no seed noise estimate. After the rerun with three seeds the extension's "
                 "interval is [-0.041, -0.012] (results.json, extension).",
    "UNC_DECISION": "The report said 'a higher' for an interval of [-0.031, +0.017]. The stated rule gives "
                    "'inconclusive', which results.json now reports.",
    "CLAIM_DIRECTION": "The accuracy claim said GSPO was higher. It now says no difference was detected, with the "
                       "interval [-0.031, +0.017] and the MDE (claims.yaml, accuracy).",
    "CLAIM_SCOPE": "A dense toy policy cannot show anything about Routing Replay in MoE training. The claim is deleted; "
                   "the paper's own result stays as a PUBLICLY DOCUMENTED quote in claims.yaml.",
    "REPORT_OVERCLAIM": "No experiment of ours could justify 'proves'. The word and its sentence are gone from "
                        "report.md, which now gives the interval [-0.031, +0.017] instead.",
}


def reference_defence(qs: list[dict]) -> str:
    """A defence answering every question of :func:`frontierlab.capstone.review.questions` on the flawed sample."""
    L = ["# Written defence — capstone: GSPO vs GRPO (sample)", ""]
    for q in qs:
        key = q["id"] if q["kind"] != "problem" else q["trigger"].split("@")[0]
        L += [f"### {q['id']}", "", f"> {q['question']}", "", DEFENCE.get(key, "See the revision log. The change "
              "is recorded in revision_log.yaml."), ""]
    return "\n".join(L)


__all__ = ["DEFENCE", "PLANTED", "apply_reruns", "reference_defence", "reference_log", "revise_by_hand", "write_sample"]
