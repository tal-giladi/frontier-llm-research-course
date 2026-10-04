"""Module 6 project: the Lineage-F integration experiment.

    python labs/module-06/project/run_project.py --variant cpu                       # free CPU (toy), ~80 minutes
    python labs/module-06/project/run_project.py --variant cpu --stages report       # analysis only
    python labs/module-06/project/run_project.py --variant main --print              # pilot rung 1 (pilot-30m)
    python labs/module-06/project/run_project.py --variant main70 --print            # pilot rung 2 (pilot-70m)
    python labs/module-06/project/run_project.py --variant lineage --lr <Baseline-0's lr> --print   # ~350M main path

Components (each passed its own correctness suite and its own lesson test): MLA (lesson 03.1), DeepSeek MTP
(06.1), mHC n = 4 (06.2) and a fine-grained MoE FFN copied from the MoE course's moelab (8 experts, top-2, one
shared expert on the toy model; 16 on the main path). The combination is all four in one BlockLM.

Stages (each resumes; finished runs are skipped):
  select   each single branch at the combination's training FLOPs, seed 0, scored on VALIDATION windows;
           the best one becomes "best single" (selection happens on validation only)
  decide   the combination (200 steps / the variant's steps), Baseline-0 at the combination's training FLOPs,
           and the best single at the combination's FLOPs, seeds 0 and 1
  attrib   factorial-lite at equal tokens, seed 0: each single (reused from lessons where possible) and the
           combination with each component left out
  report   the decision on the TEST windows (once), interactions, costs, and the run-card checks to run
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m06  # noqa: E402

COMPONENTS = ("mla", "mtp-ds", "mhc", "moe")
MARGIN = 0.0          # the decision rule's threshold in nats (stated in the project's contract)


def comps(variant: str) -> list[str]:
    return [("moe16" if c == "moe" and variant == "lineage" else c) for c in COMPONENTS]


def combo(variant: str) -> str:
    return "+".join(comps(variant))


def leave_out(variant: str, c: str) -> str:
    return "+".join(x for x in comps(variant) if x != c)


def plan(variant: str, vocab: int) -> dict:
    """Every run of the project: (arm, steps or None, seeds, stage)."""
    v = m06.VARIANTS[variant]
    cmb = combo(variant)
    eq = {c: m06.equal_flops_steps(cmb, c, variant, vocab) for c in [*comps(variant), "b0"]}
    runs = [(c, eq[c], [0], "select") for c in comps(variant)]
    runs += [(cmb, None, [0, 1], "decide"), ("b0", eq["b0"], [0, 1], "decide")]
    runs += [(c, None, [0], "attrib") for c in comps(variant)]
    runs += [(leave_out(variant, c), None, [0], "attrib") for c in comps(variant)]
    return {"runs": runs, "eq": eq, "steps": v["steps"], "combo": cmb}


def best_single(variant: str, p: dict, kw: dict, lr=None) -> tuple[str, dict]:
    scores = {}
    for c in comps(variant):
        run = m06.run_dir(variant, c, 0, lr, p["eq"][c])
        if (run / "checkpoint.pt").exists():
            scores[c] = float(np.mean(m06.eval_losses(run, **kw)))
    return (min(scores, key=scores.get) if scores else None), scores


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(m06.VARIANTS), default="cpu")
    ap.add_argument("--stages", nargs="*", default=["select", "decide", "attrib", "report"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--print", action="store_true", help="print the plan and the commands, run nothing")
    a = ap.parse_args()
    vocab = m06.vocab_size() if a.variant != "lineage" else 32768
    p = plan(a.variant, vocab)
    v = m06.VARIANTS[a.variant]
    kw = dict(n=v["eval_n"], T=v["eval_T"], device=a.device)
    if a.print or "report" in a.stages:
        print(f"Lineage-F, variant {a.variant}: combination = {p['combo']}")
        for arm in ("b0", *comps(a.variant), p["combo"]):
            cfg = m06.arm_config(arm, a.variant, vocab)
            from frontierlab.blocks import accounting
            pc = accounting.param_counts(cfg)
            print(f"  {arm:28s} total {pc['total'] / 1e6:8.2f}M  active non-emb {pc['active_non_embedding'] / 1e6:7.2f}M  "
                  f"train {m06.flops_per_token(arm, a.variant, vocab) / 1e9:6.3f} GFLOP/token")
        print("  equal-FLOPs steps (vs the combination at", p["steps"], "steps):", p["eq"])
    if a.print:
        for arm, steps, seeds, stage in p["runs"]:
            for s in seeds:
                print(f"[{stage}] {arm} seed {s} steps {steps or p['steps']}")
        return
    for arm, steps, seeds, stage in p["runs"]:
        if stage not in a.stages:
            continue
        for s in seeds:
            m06.train_arm(a.variant, arm, s, steps=steps, lr=a.lr, device=a.device, max_minutes=a.max_minutes,
                          question=f"Module 6 project (Lineage-F): {stage}, {arm}")
    best, sel = best_single(a.variant, p, dict(kw), a.lr)
    if "decide" in a.stages and best is not None:
        for s in (0, 1):
            m06.train_arm(a.variant, best, s, steps=p["eq"][best], lr=a.lr, device=a.device, max_minutes=a.max_minutes,
                          question=f"Module 6 project (Lineage-F): decide, best single {best}")
    if "report" not in a.stages:
        return
    print("\n[select] single branches at the combination's training FLOPs, seed 0, VALIDATION windows:")
    for c, x in sel.items():
        print(f"  {c:8s} {x:.4f}  ({p['eq'][c]} steps)")
    print(f"  best single: {best}")
    test = dict(kw, split="test")
    arms = {"combination": (p["combo"], None), "b0 (equal FLOPs)": ("b0", p["eq"]["b0"]),
            f"best single {best} (equal FLOPs)": (best, p["eq"][best])}
    per_seed, per_window = {}, {}
    for name, (arm, steps) in arms.items():
        runs = [m06.run_dir(a.variant, arm, s, a.lr, steps) for s in (0, 1)]
        runs = [r for r in runs if (r / "checkpoint.pt").exists()]
        if not runs:
            continue
        losses = [m06.eval_losses(r, **test) for r in runs]
        per_seed[name] = [float(np.mean(x)) for x in losses]
        per_window[name] = np.mean(losses, axis=0)
    print("\n[decide] TEST windows (used once), seeds 0 and 1, all at the combination's training FLOPs:")
    for name, xs in per_seed.items():
        print(f"  {name:36s} {' '.join(f'{x:.4f}' for x in xs)}  mean {np.mean(xs):.4f}")
    names = list(per_seed)
    res = {}
    if len(names) == 3:
        for other in names[1:]:
            r = res[other] = m06.compare(per_window["combination"], per_window[other])
            d = [x - y for x, y in zip(per_seed["combination"], per_seed[other])]
            print(f"  combination - {other}: {m06.fmt(r)}   per seed {' '.join(f'{z:+.4f}' for z in d)}")
        nf = m06.noise_floor(per_seed[names[1]], len(per_seed[names[1]]))
        print(f"  noise floor (b0 seeds): std {nf['seed_std']:.4f}, MDE {nf['mde']:.4f}")
        r = res[names[2]]
        d = [x - y for x, y in zip(per_seed["combination"], per_seed[names[2]])]
        if r["ci"][1] < -MARGIN and all(z < 0 for z in d):
            verdict = "ADOPT the combination as the chosen architecture (beats its best single branch at equal FLOPs)"
        elif r["ci"][0] > MARGIN:
            verdict = f"DO NOT ADOPT: the combination is worse than its best single branch ({best}) at equal FLOPs"
        else:
            verdict = f"INCONCLUSIVE: carry the simpler choice ({best} alone, or Baseline-0 if it does not beat it)"
        print(f"\nDecision rule: {verdict}")
    print("\n[attrib] factorial-lite at equal tokens, seed 0, VALIDATION windows:")
    base = m06.run_dir(a.variant, "b0", 0, a.lr)
    cmb = m06.run_dir(a.variant, p["combo"], 0, a.lr)
    if (base / "checkpoint.pt").exists() and (cmb / "checkpoint.pt").exists():
        lb, lc = np.array(m06.eval_losses(base, **kw)), np.array(m06.eval_losses(cmb, **kw))
        print(f"  b0 {lb.mean():.4f}   combination {lc.mean():.4f}")
        singles = {}
        for c in comps(a.variant):
            rs, rl = m06.run_dir(a.variant, c, 0, a.lr), m06.run_dir(a.variant, leave_out(a.variant, c), 0, a.lr)
            if not ((rs / "checkpoint.pt").exists() and (rl / "checkpoint.pt").exists()):
                continue
            ls, ll = np.array(m06.eval_losses(rs, **kw)), np.array(m06.eval_losses(rl, **kw))
            singles[c] = ls.mean()
            alone, inside = m06.compare(ls, lb), m06.compare(lc, ll)
            print(f"  {c:8s} effect alone (single - b0) {m06.fmt(alone)};  inside (combination - leave-out) "
                  f"{m06.fmt(inside)};  total interaction {inside['mean_diff'] - alone['mean_diff']:+.4f}")
        if singles:
            additive = lb.mean() + sum(x - lb.mean() for x in singles.values())
            print(f"  additive prediction for the combination {additive:.4f}; measured {lc.mean():.4f} "
                  f"(difference {lc.mean() - additive:+.4f})")
    print("\ncosts (measured wall-clock of seed 0, seconds):")
    for arm, steps in [(p["combo"], None), ("b0", p["eq"]["b0"]), *((c, p["eq"][c]) for c in comps(a.variant))]:
        print(f"  {arm:28s} {m06.wallclock(m06.run_dir(a.variant, arm, 0, a.lr, steps)):8.0f}")
    out = {"best_single": best, "select_val": sel, "test_per_seed": per_seed,
           "decide": {k: {"mean_diff": r["mean_diff"], "ci": r["ci"]} for k, r in res.items()}}
    path = m06.ROOT / a.variant / "lineage_f_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
