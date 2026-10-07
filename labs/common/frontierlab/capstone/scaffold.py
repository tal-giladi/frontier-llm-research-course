"""Capstone scaffold for one claim, end to end: QK-Clip vs unclipped Muon (reproduction) and vs QK-norm (extension).

    python -m frontierlab.capstone.scaffold --out runs/m20/capstone-qkclip              # free CPU, about 10 minutes
    python -m frontierlab.capstone.scaffold --variant main --print                      # main-path commands (PROJECTED)

What it does, in order (lesson 20.1):

1. **Correctness checks first** (:func:`correctness_checks`): in float64, QK-Clip caps every head's recomputed maximum
   logit at exactly ``min(S_max, tau)`` on the same layer inputs; QK-Clip refuses a model with QK-norm (which would
   undo it); the no-QK-norm model passes the causal and cached-decode checks. Results do not count unless all pass.
2. **Writes the package skeleton before any run**: ``claim.yaml`` (arms, roles, seeds, comparisons and the variables
   each changes, axis, margin, tuning budget) and ``contract.md`` (the experiment contract, filled).
3. **Runs the arms** through ``python -m frontierlab.optim.train`` (the unmodified course loop with the Module 7
   wrapper), so every run writes its own ``run_card.yaml`` with its parent, and exact resume holds:

   ``muon-noqk``   Muon, no QK-norm, no clip                      baseline (vanilla Muon, K2 Appendix D's reference)
   ``muon-clip``   Muon, no QK-norm, QK-Clip at tau              reproduction (K2 Appendix D: negligible loss impact?)
   ``muon-qknorm`` Muon with QK-norm (DeepSeek-V4's choice)      extension (the comparison neither report ran)

   tau follows a rule fixed before the clip runs: ``round(TAU_FRACTION x S_max)`` of the seed-0 baseline run at its
   last step, so the clip binds for the last part of training (Appendix D's "aggressive" tau played the same role).
4. **Evaluates** each final checkpoint on the same fixed held-out windows and computes, per comparison, the paired
   hierarchical bootstrap over seeds and windows, the paired t-interval over seeds, and the decision by the rule.
   Secondary: run maximum logit, clipped head-updates, loss spikes (lesson 07.5's detector).
5. **Writes** ``results.json``, a draft ``claims.yaml`` and ``report.md`` whose sentences follow the decisions, and
   runs :func:`frontierlab.capstone.package.check_package` on the result.

The interval and decision functions can be replaced (the lab passes the learner's own).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import time
from pathlib import Path, PurePosixPath

import numpy as np
import torch
import yaml

from frontierlab.capstone import uncertainty as U
from frontierlab.capstone.package import check_package, run_name
from frontierlab.data.loader import TokenData
from frontierlab.evals.heldout import window_losses
from frontierlab.metrics.jsonl import read_jsonl

VARIANTS = {   # preset, steps, batch, seq, lr, extra loop args, eval windows, eval T
    "cpu": ("toy", 200, 8, 128, 1e-2, ["--eval-every", "200", "--eval-windows", "32", "--ckpt-every", "100",
                                         "--log-every", "1"], 256, 128),
    "t4": ("pilot-10m", 2000, 32, 512, 1e-2, ["--dtype", "fp32", "--eval-every", "500", "--ckpt-every", "250",
                                               "--log-every", "5"], 256, 512),
    "main": ("pilot-30m", 4000, 64, 1024, 1e-2, ["--dtype", "bf16", "--eval-every", "500", "--ckpt-every", "500",
                                                 "--log-every", "10", "--peak", "H100-SXM"], 256, 1024),
}
TAU_FRACTION = 0.6
MARGIN = 0.02            # nats of held-out loss: a difference smaller than this does not matter for the decision
EXTERNAL_PARENT = "m07-l72-muon-noqk"
ARMS = {
    "muon-noqk": {"role": "baseline", "flags": ["--optimizer", "muon", "--qk-norm", "off"]},
    "muon-clip": {"role": "reproduction", "flags": ["--optimizer", "muon", "--qk-norm", "off", "--qk-clip", "{tau}"]},
    "muon-qknorm": {"role": "extension", "flags": ["--optimizer", "muon", "--qk-norm", "on"]},
}
COMPARISONS = [
    {"name": "reproduction", "a": "muon-clip", "b": "muon-noqk", "changed": ["optim.qk_clip"]},
    {"name": "extension", "a": "muon-qknorm", "b": "muon-clip",
     "changed": ["optim.qk_clip", "optim.qk_norm", "config.qk_norm"]},
]


# --------------------------------------------------------------------------- correctness

def correctness_checks() -> dict:
    """The three checks the contract requires before any result counts. Returns {name: value, "passed": bool}."""
    from frontierlab.model import LM, toy
    from frontierlab.optim.qkclip import LogitMonitor, QKClip, head_max_logits
    from frontierlab.testing import cache_agreement, causal_check
    torch.manual_seed(0)
    model = LM(toy(vocab_size=61).with_(qk_norm=False)).double()
    for n, p in model.named_parameters():
        if "q_proj" in n or "k_proj" in n:
            p.data.mul_(8.0)                                        # make logits large enough to clip
    x = torch.randint(0, 61, (2, 12), generator=torch.Generator().manual_seed(0))
    mods = [l.self_attn for l in model.model.layers]
    inputs = {}
    hooks = [m.register_forward_pre_hook(lambda m, a, i=i: inputs.__setitem__(i, (a[0].detach(), a[1])))
             for i, m in enumerate(mods)]
    model(x)
    for h in hooks:
        h.remove()
    before = [head_max_logits(m, *inputs[i]) for i, m in enumerate(mods)]
    tau = float(torch.stack(before).median())
    clip = QKClip(model, tau, monitor=LogitMonitor(model))
    clip.monitor.smax = {i: b for i, b in enumerate(before)}
    clip()
    after = [head_max_logits(m, *inputs[i]) for i, m in enumerate(mods)]
    cap_err = max(float((a - torch.clamp(b, max=tau)).abs().max()) for b, a in zip(before, after))
    try:
        QKClip(LM(toy(vocab_size=61)), 10.0)
        refuses = False
    except ValueError:
        refuses = True
    torch.manual_seed(1)
    plain = LM(toy(vocab_size=61).with_(qk_norm=False))
    out = {"qkclip_cap_max_abs_err": cap_err, "qkclip_clipped_heads": int(clip.last["clipped_heads"]),
           "qkclip_refuses_qk_norm": refuses, "causal_max_abs_diff": float(causal_check(plain, 61)),
           "cache_max_abs_diff": float(cache_agreement(plain, 61, chunk=1))}
    out["passed"] = bool(cap_err < 1e-9 and out["qkclip_clipped_heads"] > 0 and refuses
                         and out["causal_max_abs_diff"] < 1e-9 and out["cache_max_abs_diff"] < 1e-9)
    return out


# --------------------------------------------------------------------------- runs

def finished(run: Path, steps: int) -> bool:
    log = run / "metrics.jsonl"
    if not log.exists():
        return False
    rows = [r for r in read_jsonl(log) if r["split"] == "train"]
    return bool(rows) and rows[-1]["step"] >= steps


def parent_of(arm: str, seed: int) -> str:
    return EXTERNAL_PARENT if ARMS[arm]["role"] == "baseline" else run_name("muon-noqk", seed)


def arm_argv(arm: str, seed: int, variant: str, tau: float | None, run: Path) -> list[str]:
    preset, steps, batch, seq, lr, extra, _, _ = VARIANTS[variant]
    flags = [f.format(tau=f"{tau:g}" if tau is not None else "TAU") for f in ARMS[arm]["flags"]]
    return ["--run", str(run), "--preset", preset, "--steps", str(steps), "--batch", str(batch), "--seq", str(seq),
            "--lr", f"{lr:g}", "--seed", str(seed), "--stability-log", "--parent", parent_of(arm, seed),
            "--question", f"capstone qkclip: arm {arm}", *flags, *extra]


def train_run(argv: list[str], run: Path, steps: int) -> float:
    from frontierlab.optim import train as optim_train
    if finished(run, steps):
        return 0.0
    t0 = time.perf_counter()
    optim_train.main(argv)
    dt = time.perf_counter() - t0
    with open(run / "wallclock.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"seconds": round(dt, 2)}) + "\n")
    return dt


def last_max_logit(run: Path) -> float:
    rows = read_jsonl(run / "stability.jsonl")
    return float(rows[-1]["max_logit"])


def choose_tau(baseline_run: Path, fraction: float = TAU_FRACTION) -> float:
    """The rule stated in the contract: round(fraction x the seed-0 baseline's maximum logit at its last step)."""
    return float(max(1, round(fraction * last_max_logit(baseline_run))))


def eval_losses(run: Path, n: int, T: int, data: str | None = None) -> list[float]:
    cache = run / f"eval_{n}x{T}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from frontierlab.optim import train as optim_train
    model = optim_train.load_model(run / "checkpoint.pt")
    losses = window_losses(model, TokenData("val", **({"root": data} if data else {})), n, T)
    cache.write_text(json.dumps(losses))
    return losses


def run_stats(run: Path) -> dict:
    from frontierlab.optim.stability import detect_spikes
    st = read_jsonl(run / "stability.jsonl")
    tr = [r for r in read_jsonl(run / "metrics.jsonl") if r["split"] == "train"]
    spikes = detect_spikes([r["step"] for r in tr], [r["loss"] for r in tr])
    wc = run / "wallclock.jsonl"
    return {"max_logit": max(float(r["max_logit"]) for r in st),
            "final_max_logit": float(st[-1]["max_logit"]),
            "clipped_head_updates": int(sum(r.get("clipped_heads") or 0 for r in st)),
            "first_clip_step": next((r["step"] for r in st if (r.get("clipped_heads") or 0) > 0), None),
            "loss_spikes": len(spikes),
            "train_seconds": sum(r["seconds"] for r in read_jsonl(wc)) if wc.exists() else None}


# --------------------------------------------------------------------------- package files

def claim_yaml(variant: str, seeds: list[int], tau: float | None) -> dict:
    preset, steps, batch, seq, lr, _, n, T = VARIANTS[variant]
    return {"claim": "qkclip", "variant": variant, "axis": "tokens", "metric": f"held-out loss, {n} windows x {T}",
            "lower_is_better": True, "margin": MARGIN, "seeds": list(seeds), "tau": tau,
            "tau_rule": f"round({TAU_FRACTION} x the seed-0 baseline's max logit at its last step)",
            "external_parents": [EXTERNAL_PARENT],
            "tuning": {a: 1 for a in ARMS},          # one learning rate (lesson 07.2's raised 1e-2) for every arm
            "arms": {a: {"role": v["role"], "flags": v["flags"]} for a, v in ARMS.items()},
            "comparisons": COMPARISONS,
            "budget": {"preset": preset, "steps": steps, "batch": batch, "seq": seq, "lr": lr,
                       "tokens_per_run": steps * batch * seq}}


CONTRACT = """# Experiment contract — capstone: QK-Clip vs unclipped Muon, and vs QK-norm

Written by `frontierlab.capstone.scaffold` before any run ({variant} variant). Edit it before you run; record any later
change in the last section.

## Question and decision

- **Question:** at equal tokens, does QK-Clip at a tau that binds change held-out loss relative to unclipped Muon by
  more than {margin} nats (reproduction of Kimi K2 Appendix D), and does it differ from QK-norm by more than that
  (extension)?
- **Decision it informs:** which logit fix the course's later Muon runs use for GQA models without MLA.

## Hypothesis

- **Hypothesis:** H1, clipping costs less than {margin} nats (equivalent within the margin). H2, QK-norm and QK-Clip
  are equivalent within the margin. H3, the unclipped baseline's maximum logit grows during training.
- **Status:** H1 is a reported effect (Kimi K2 Appendix D, Figure 12, at 0.5B activated / 3B total and tau = 30);
  H2 may not appear at this scale (no report tests it); H3 is a reported effect (K2 section 2.1, Figure 2), and
  lesson 07.2 measured it at toy scale.

## Baseline

- **Baseline run card:** `runs/muon-noqk-s<seed>` in this package; parent `{parent}` (lesson 07.2's arm, same setting).
- **How it was tuned:** not re-tuned: every arm uses lesson 07.2's raised learning rate {lr:g} (one trial each, the
  same tuning budget for every arm, recorded as `tuning` in claim.yaml).

## What changes and what is held fixed

- **Changed variable:** the logit fix: none, QK-Clip at tau (rule: {tau_rule}), or QK-norm.
- **Held fixed:** Data-v0 (hashes in each run card), preset {preset}, {steps} steps x {batch} x {seq} tokens, Muon
  settings, schedule, seeds {seeds} (same initialisation and data order per seed across arms), evaluation on the same
  {n} validation windows of {T} tokens, software versions.

## Comparison axis

Equal tokens, because the claim is about loss at a given amount of training. It does not answer which fix is cheaper
per step (QK-Clip adds a logit probe; QK-norm adds two RMSNorms per layer); report the measured step time as a
secondary number only.

## Budget

- {variant} variant: {runs} runs of {tokens_per_run:,} tokens each. Main path PROJECTED about 20 GPU-hours on 1x H100
  (pilot-30m, pilot-70m and Baseline-0 rungs; formula in `frontierlab.capstone.claims`).
- No teacher, judge or generator compute.
- Tuning budget per arm: one learning rate (the same for all).

## Metrics and decision rule

- **Primary metric:** held-out loss per window; uncertainty by the paired hierarchical bootstrap over seeds and windows
  (95%), with the paired t-interval over seed means reported alongside.
- **Secondary metrics:** run maximum logit, clipped head-updates and the first step a clip fired, loss spikes.
- **Noise floor:** the baseline's seed standard deviation and the minimum detectable effect for {n_seeds} seeds.
- **Decision rule:** equivalent if the 95% interval of a - b lies inside [-{margin}, +{margin}] nats; a lower / a higher
  if it excludes zero otherwise; inconclusive if not. If the clip never fires, the reproduction is "not tested".

## Correctness checks (must pass before results count)

- [x] QK-Clip caps recomputed per-head logits at min(S_max, tau) in float64; QK-Clip refuses a QK-norm model
- [x] causal and cached-decode checks pass for the no-QK-norm model
- [x] budget parity verified from the run cards (`frontierlab.capstone.package.check_package`)

## Fallback evidence

If the clip never fires at this scale: lesson 07.2's induced tau = 15 runs and Kimi K2's Figure 2 and Figure 12,
labelled "analysis of published results", not reproduction.

## Limits of the conclusion

A 1.8M-parameter dense model (toy) trained for {steps} steps on Data-v0, not a 3B MoE trained on trillions of
tokens; GQA, where QK-Clip scales only the query rows (a course choice, INFERENCE from K2's MLA rule); one learning
rate; logits around 10, not 1,000. A different answer at the pilot-70m rung would change the conclusion.

## Changes after results

None yet.
"""


def write_contract(pkg: Path, variant: str, seeds: list[int], tau: float | None) -> None:
    preset, steps, batch, seq, lr, _, n, T = VARIANTS[variant]
    text = CONTRACT.format(variant=variant, margin=MARGIN, parent=EXTERNAL_PARENT, lr=lr, preset=preset, steps=steps,
                           batch=batch, seq=seq, seeds=seeds, n=n, T=T, runs=len(ARMS) * len(seeds),
                           tokens_per_run=steps * batch * seq, n_seeds=len(seeds),
                           tau_rule=f"round({TAU_FRACTION} x the seed-0 baseline's max logit at its last step)")
    (pkg / "contract.md").write_text(text, encoding="utf-8")


WORDS = {"equivalent": "equivalent within the margin", "a_lower": "lower", "a_higher": "higher",
         "inconclusive": "not distinguishable (inconclusive)"}
DIRECTION = {"equivalent": "equivalent", "a_lower": "lower", "a_higher": "higher", "inconclusive": "no-difference-detected"}


def draft_claims(results: dict, tau: float, variant: str) -> list[dict]:
    """One MEASURED claim per comparison, worded by the decision; one PUBLICLY DOCUMENTED restatement."""
    out = []
    for r in results["comparisons"]:
        lo, hi = r["ci"]
        out.append({"id": r["name"], "comparison": r["name"], "label": "MEASURED", "scope": "course",
                    "direction": DIRECTION[r["decision"]],
                    "text": f"At course scale ({variant} variant), {r['a']} minus {r['b']} held-out loss was "
                            f"{r['mean_diff']:+.4f} nats [{lo:+.4f}, {hi:+.4f}] over {r['n_seeds']} seeds: "
                            f"{WORDS[r['decision']]} (margin {MARGIN})."})
    out.append({"id": "source", "label": "PUBLICLY DOCUMENTED", "scope": "frontier",
                "source": "Kimi K2 (arXiv 2507.20534) Appendix D, Figure 12",
                "text": "Kimi K2 reports that QK-Clip at tau = 30 had negligible impact on loss for 0.5B-activated, "
                        "3B-total MoE models."})
    return out


def draft_report(results: dict, stats: dict, checks: dict, tau: float, variant: str) -> str:
    L = ["# Capstone report — QK-Clip vs unclipped Muon, and vs QK-norm", "",
         f"Draft written by the scaffold ({variant} variant). Rewrite it in your own words; keep every number's interval.",
         "", "## Result", ""]
    for r in results["comparisons"]:
        L.append(f"- **{r['name']}** ({r['a']} - {r['b']}): {r['mean_diff']:+.4f} nats, 95% interval "
                 f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] ({r['method']}); seed t-interval "
                 f"[{r['t_ci'][0]:+.4f}, {r['t_ci'][1]:+.4f}]. Decision: {r['decision']}.")
    nf = results["noise_floor"]
    L += ["", f"Noise floor: baseline seed std {nf['seed_std']:.4f} nats, minimum detectable effect {nf['mde']:.4f} "
              f"with {nf['n_seeds']} seeds.", "", "## Secondary measurements", "",
          "| run | max logit | final max logit | clipped head-updates | first clip step | loss spikes |", "|---|---|---|---|---|---|"]
    for name, s in stats.items():
        L.append(f"| {name} | {s['max_logit']:.2f} | {s['final_max_logit']:.2f} | {s['clipped_head_updates']} | "
                 f"{s['first_clip_step']} | {s['loss_spikes']} |")
    L += ["", f"tau = {tau:g} by the stated rule. Correctness checks: {'all passed' if checks['passed'] else 'FAILED'}.", "",
          "## Limits", "",
          "Toy scale (1.8M parameters, dense GQA), one learning rate, Data-v0, a few seeds; maximum logits of order 10, "
          "not 1,000. Nothing here is evidence about MoE models at Kimi K2's scale.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- main

def run(out: str | Path = "runs/m20/capstone-qkclip", variant: str = "cpu", seeds=(0, 1, 2),
        interval=None, decide=None, device: str = "cpu", steps: int | None = None, data: str | None = None,
        eval_n: int | None = None) -> dict:
    """Run the whole capstone for the QK-Clip claim and write the package. ``interval(a, b)`` takes two
    (seeds, windows) arrays and returns {"mean_diff", "ci", "method", ...}; ``decide(mean, lo, hi, margin)``."""
    interval = interval or U.hierarchical_bootstrap
    decide = decide or U.decide
    pkg = Path(out)
    (pkg / "runs").mkdir(parents=True, exist_ok=True)
    seeds = list(seeds)
    if steps is not None:                                      # tests: a shorter run of the same design
        v = list(VARIANTS[variant])
        v[1] = steps
        v[5] = [x if x not in (str(VARIANTS[variant][1]), "100") else str(steps) for x in v[5]]
        if eval_n:
            v[6] = eval_n
        VARIANTS["_short"] = tuple(v)
        variant = "_short"
    checks = correctness_checks()
    (pkg / "checks.json").write_text(json.dumps(checks, indent=1))
    if not checks["passed"]:
        raise SystemExit(f"correctness checks failed, results would not count: {checks}")
    claim = claim_yaml(variant, seeds, None)
    (pkg / "claim.yaml").write_text(yaml.safe_dump(claim, sort_keys=False))
    write_contract(pkg, variant, seeds, None)
    steps_ = VARIANTS[variant][1]
    extra = ["--device", device] + (["--data", str(data)] if data else [])
    base = "muon-noqk"
    for s in seeds:                                            # the baseline first: tau comes from its seed 0
        r = pkg / "runs" / run_name(base, s)
        train_run(arm_argv(base, s, variant, None, r) + extra, r, steps_)
    tau = choose_tau(pkg / "runs" / run_name(base, seeds[0]))
    claim["tau"] = tau
    (pkg / "claim.yaml").write_text(yaml.safe_dump(claim, sort_keys=False))
    for arm in ARMS:
        if arm == base:
            continue
        for s in seeds:
            r = pkg / "runs" / run_name(arm, s)
            train_run(arm_argv(arm, s, variant, tau, r) + extra, r, steps_)
    _, _, _, _, _, _, n, T = VARIANTS[variant]
    losses = {a: np.array([eval_losses(pkg / "runs" / run_name(a, s), n, T, data) for s in seeds]) for a in ARMS}
    comps = []
    for c in COMPARISONS:
        a, b = losses[c["a"]], losses[c["b"]]
        h = interval(a, b)
        t = U.seed_t_interval(a.mean(1), b.mean(1))
        lo, hi = h["ci"]
        comps.append({"name": c["name"], "a": c["a"], "b": c["b"], "metric": claim["metric"], "n_seeds": len(seeds),
                      "per_seed": {"a": a.mean(1).tolist(), "b": b.mean(1).tolist()},
                      "mean_diff": float(h["mean_diff"]), "ci": [float(lo), float(hi)], "method": h["method"],
                      "t_ci": [float(x) for x in t["ci"]],
                      "decision": decide(float(h["mean_diff"]), float(lo), float(hi), MARGIN)})
    nf = U.noise_floor(losses[base].mean(1))
    stats = {run_name(a, s): run_stats(pkg / "runs" / run_name(a, s)) for a in ARMS for s in seeds}
    results = {"claim": "qkclip", "tau": tau, "margin": MARGIN, "comparisons": comps,
               "noise_floor": {"arm": base, **nf}, "secondary": stats, "checks": checks,
               "machine": {"torch": torch.__version__, "threads": torch.get_num_threads(), "device": device}}
    (pkg / "results.json").write_text(json.dumps(results, indent=1))
    (pkg / "claims.yaml").write_text(yaml.safe_dump(draft_claims(results, tau, variant), sort_keys=False, width=1000))
    (pkg / "report.md").write_text(draft_report(results, stats, checks, tau, variant), encoding="utf-8")
    results["problems"] = [str(p) for p in check_package(pkg)]
    return results


def print_commands(variant: str, seeds=(0, 1, 2), out: str = "runs/m20/capstone-qkclip") -> list[str]:
    """The GPU commands of a variant (the main path is not run in this build; Module 20 pilot)."""
    out = Path(out).as_posix()
    lines = [f"# {variant}: correctness checks and package skeleton, then the baseline seeds, then tau, then the rest",
             f"python -m frontierlab.capstone.scaffold --variant {variant} --out {PurePosixPath(out)}-{variant} --device cuda"]
    for arm in ARMS:
        for s in seeds:
            r = f"{out}-{variant}/runs/{run_name(arm, s)}"
            lines.append("python -m frontierlab.optim.train " + " ".join(shlex.quote(x) for x in arm_argv(arm, s, variant, None, PurePosixPath(r)))
                         + " --device cuda" + ("   # TAU: the rule's value, printed by the scaffold after the baseline seeds" if arm == "muon-clip" else ""))
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/m20/capstone-qkclip")
    ap.add_argument("--variant", choices=["cpu", "t4", "main"], default="cpu")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--print", action="store_true", help="print the commands of the variant and exit")
    a = ap.parse_args(argv)
    if a.print:
        print("\n".join(print_commands(a.variant, a.seeds, a.out)))
        return None
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    t0 = time.perf_counter()
    res = run(a.out, a.variant, a.seeds, device=a.device)
    for c in res["comparisons"]:
        print(f"{c['name']:12s} {c['a']} - {c['b']}: {c['mean_diff']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]"
              f"  t [{c['t_ci'][0]:+.4f}, {c['t_ci'][1]:+.4f}]  -> {c['decision']}")
    print(f"tau {res['tau']:g}; noise floor {res['noise_floor']}")
    print("package problems:", res["problems"] or "none")
    print(f"total {time.perf_counter() - t0:.0f} s")
    return res


if __name__ == "__main__":
    main()
