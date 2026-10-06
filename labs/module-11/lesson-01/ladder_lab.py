"""Lab 11.1: an iso-FLOP ladder, Chinchilla's approaches 2 and 3 on it, a repetition experiment, published numbers.

    python labs/module-11/lesson-01/ladder_lab.py published                 # no training: your TODOs on published models
    python labs/module-11/lesson-01/ladder_lab.py isoflop                   # the ladder (CPU: see the lesson for the time)
    python labs/module-11/lesson-01/ladder_lab.py repeat                    # unique data vs repeated data at equal tokens
    python labs/module-11/lesson-01/ladder_lab.py isoflop --variant main --print     # main-path commands (GPU)

``isoflop`` trains every (budget, rung) of the variant's grid through ``frontierlab.scaling.train`` (finished runs are
skipped, interrupted ones resume exactly), then fits: approach 2 (your ``isoflop_vertex`` per budget, then
N_opt ∝ C^a) with total and with non-embedding parameters, approach 3 (``frontierlab.scaling.fit.fit_parametric``) on
the two smaller budgets, checked on the largest (held out). Results: ``runs/m11/l111/<variant>/results.json``; lessons
11.2 and 11.3 reuse these runs.

``repeat`` trains one rung for the same number of tokens drawn from 100%, 1/4, 1/16 and 1/64 of the training split
(1, 4, 16 and 64 epochs), two seeds each, and compares the measured loss penalty with Muennighoff et al.'s effective data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

from frontierlab.scaling import fit as sfit  # noqa: E402
from frontierlab.scaling import ladder, laws  # noqa: E402
from frontierlab.stats import summary  # noqa: E402

DATA_CPU = common.REPO / "labs" / "common" / "data" / "m11-v1024"
VARIANTS = {
    "cpu": dict(data=str(DATA_CPU), rungs=["m11-r1", "m11-r2", "m11-r3", "m11-r4", "m11-r5", "m11-r6"], budgets=[1e12, 3e12, 1e13],
                seq=128, batch=16, lr=3e-3, windows=64, device="cpu", vocab=1024, max_tokens=3.0e7,
                repeat_rung="m11-r3", repeat_steps=1500, repeat_fracs=[1.0, 0.25, 1 / 16, 1 / 64], seeds=[0, 1]),
    "t4": dict(data=str(DATA_CPU), rungs=["m11-r3", "m11-r4", "m11-r5", "m11-r6", "m11-r7"], budgets=[1e13, 3e13, 1e14],
               seq=256, batch=32, lr=3e-3, windows=128, device="cuda", vocab=1024, max_tokens=3.6e7,
               repeat_rung="m11-r5", repeat_steps=2000, repeat_fracs=[1.0, 0.25, 1 / 16, 1 / 64], seeds=[0, 1]),
    "main": dict(data="labs/common/data/v0-main", rungs=["pilot-10m", "pilot-30m", "pilot-70m", "baseline0", "m11-200m"],
                 budgets=[1e17, 3e17, 1e18], seq=1024, batch=32, lr=3e-3, windows=256, device="cuda", dtype="bf16",
                 vocab=32768, max_tokens=4.0e9, repeat_rung="pilot-30m", repeat_steps=8000,
                 repeat_fracs=[1.0, 0.25, 1 / 16, 1 / 64], seeds=[0, 1]),
}


def out_dir(variant: str) -> Path:
    return common.RUNS / "l111" / variant


def isoflop_plan(v: dict) -> list[dict]:
    plan = ladder.plan_isoflop(v["budgets"], v["rungs"], seq=v["seq"], batch=v["batch"], vocab_size=v["vocab"])
    return [p for p in plan if p["tokens"] <= v["max_tokens"]]          # never repeat data inside the ladder


def run_isoflop(v: dict, variant: str, print_only: bool = False) -> list[dict]:
    runs = []
    for p in isoflop_plan(v):
        run = out_dir(variant) / f"{p['preset']}-c{p['budget']:.0e}"
        args = common.run_args(p, v, run, extra=["--data", v["data"], "--question", "iso-FLOP ladder (11.1)"])
        r = common.ensure_run(run, args, print_only=print_only)
        if r is not None:
            runs.append({**r, "budget": p["budget"], "preset": p["preset"]})
    return runs


WINDOW = 3          # approach 2: fit each parabola to the 3 lowest-loss runs of the budget (the neighbourhood of the minimum)
MIN_TPP = 4.0       # approach 3: drop runs with fewer than 4 training tokens per parameter (a few hundred steps: still
                    # optimisation-limited, outside the regime the law describes). Both rules were set after the build's
                    # all-points fit; lesson 11.3 tests them on a run nobody had seen.


def approach2(lab, runs: list[dict], key: str, window: int | None = None) -> dict:
    mins = []
    for b in sorted({r["budget"] for r in runs}):
        g = sorted([r for r in runs if r["budget"] == b], key=lambda r: r["loss"])
        if window:
            g = g[:window]
        try:
            n_opt = lab.isoflop_vertex([r[key] for r in g], [r["loss"] for r in g])
        except ValueError:
            n_opt = float("nan")
        Ns = [r[key] for r in g]
        mins.append({"budget": b, "N_opt": n_opt, "edge": not (min(Ns) <= n_opt <= max(Ns)), "best_run": g[0]["run"]})
    ok = [m for m in mins if np.isfinite(m["N_opt"])]
    alloc = sfit.fit_allocation([m["budget"] for m in ok], [m["N_opt"] for m in ok]) if len(ok) >= 2 else None
    return {"minima": mins, "a": alloc["a"] if alloc else float("nan")}


def approach3(runs: list[dict], min_tpp: float = 0.0) -> dict:
    sel = [r for r in runs if r["D"] / r["N_total"] >= min_tpp]
    budgets = sorted({r["budget"] for r in sel})
    N = np.array([r["N_total"] for r in sel], dtype=float)
    D = np.array([r["D"] for r in sel], dtype=float)
    L = np.array([r["loss"] for r in sel])
    test = np.array([r["budget"] == budgets[-1] for r in sel])
    held = sfit.holdout(N, D, L, test)
    full = sfit.fit_parametric(N, D, L)
    return {"runs": [r["run"] for r in sel], "test": [r["run"] for r, t in zip(sel, test) if t], "holdout": held, "fit_all": full,
            "test_rows": [(r["preset"], p, r["loss"]) for r, p in zip([r for r, t in zip(sel, test) if t], held["pred"])],
            "n_train": int((~test).sum()), "budget_test": budgets[-1]}


def cmd_isoflop(lab, a, v):
    runs = run_isoflop(v, a.variant, a.print)
    if a.print:
        return
    print("\nIso-FLOP ladder (validation loss, Eval v0 windows):")
    print(f"{'budget':>8} {'rung':>8} {'N_total':>9} {'N_nonemb':>9} {'tokens':>10} {'tok/param':>9} {'loss':>7}")
    for r in sorted(runs, key=lambda r: (r["budget"], r["N_total"])):
        print(f"{r['budget']:8.0e} {r['preset']:>8} {r['N_total']:9d} {r['N_nonemb']:9d} {r['D']:10.3g} {r['D'] / r['N_total']:9.1f} {r['loss']:7.4f}")
    res = {"runs": runs, "window": WINDOW, "min_tpp": MIN_TPP}
    for window in (None, WINDOW):
        for key in ("N_total", "N_nonemb"):
            a2 = approach2(lab, runs, key, window)
            name = f"approach2_{key}" + (f"_w{window}" if window else "")
            res[name] = a2
            print(f"\nApproach 2, {key}, {'all runs' if not window else f'the {window} lowest per budget'}: "
                  f"exponent a of N_opt ∝ C^a = {a2['a']:.3f}")
            for m in a2["minima"]:
                print(f"  C = {m['budget']:.0e}: N_opt = {m['N_opt']:.3g}{'  (outside the fitted sizes)' if m['edge'] else ''}; "
                      f"best run {m['best_run']}")
    for tpp in (0.0, MIN_TPP):
        a3 = approach3(runs, tpp)
        f, held, full = a3["holdout"]["fit"], a3["holdout"], a3["fit_all"]
        res[f"approach3_tpp{tpp:g}"] = a3
        print(f"\nApproach 3, runs with >= {tpp:g} tokens/param: fit on {a3['n_train']} runs below {a3['budget_test']:.0e}: "
              f"E = {f['E']:.3f}, alpha = {f['alpha']:.3f}, beta = {f['beta']:.3f}, in-sample RMS(log) {f['rms_log']:.4f}")
        print(f"  held-out budget {a3['budget_test']:.0e}: mean |error| = {held['mae']:.4f} nats, max {held['max_abs']:.4f}")
        for preset, p, m in a3["test_rows"]:
            print(f"    {preset:>8}: predicted {p:.4f}, measured {m:.4f}")
        print(f"  on all {len(a3['runs'])} runs: alpha = {full['alpha']:.3f}, beta = {full['beta']:.3f}, "
              f"beta/(alpha+beta) = {full['beta'] / (full['alpha'] + full['beta']):.3f}")
    res["holdout"] = res[f"approach3_tpp{MIN_TPP:g}"]["holdout"]          # what lesson 11.3's checklist reads
    res["fit_all"] = res[f"approach3_tpp{MIN_TPP:g}"]["fit_all"]
    ladder.save_json(res, out_dir(a.variant) / "results.json")
    print(f"\nwrote {out_dir(a.variant) / 'results.json'}")


def cmd_repeat(lab, a, v):
    from frontierlab.data.loader import TokenData
    n_train = len(TokenData("train", v["data"])) if not a.print else int(v["max_tokens"])
    D = v["repeat_steps"] * v["batch"] * v["seq"]
    rows = []
    for frac in v["repeat_fracs"]:
        U = int(min(n_train, D) * frac) if frac < 1 else n_train
        for seed in v["seeds"]:
            run = out_dir(a.variant) / f"repeat-{v['repeat_rung']}-u{U}-s{seed}"
            p = {"preset": v["repeat_rung"], "steps": v["repeat_steps"]}
            args = common.run_args(p, v, run, seed=seed, extra=["--data", v["data"], "--question", "repetition (11.1)"])
            if frac < 1:
                args = ["--unique-tokens", str(U)] + args
            r = common.ensure_run(run, args, print_only=a.print)
            if r is not None:
                rows.append({"U": U, "epochs": D / U, "seed": seed, "loss": r["loss"]})
    if a.print:
        return
    print(f"\nRepetition at equal tokens: {v['repeat_rung']}, D = {D:,} tokens, seeds {v['seeds']}")
    base = summary([r["loss"] for r in rows if r["epochs"] <= 1.0])
    print(f"{'unique U':>10} {'epochs':>7} {'loss (mean of seeds)':>21} {'- unique':>9} {'D_eff/D (Muennighoff)':>22}")
    out = []
    for U in sorted({r["U"] for r in rows}, reverse=True):
        g = [r for r in rows if r["U"] == U]
        s = summary([r["loss"] for r in g])
        d_eff = float(lab.effective_data(U, D)) / D
        out.append({"U": U, "epochs": g[0]["epochs"], "loss": s["mean"], "std": s["std"], "penalty": s["mean"] - base["mean"], "d_eff_frac": d_eff})
        print(f"{U:10,d} {g[0]['epochs']:7.2f} {s['mean']:12.4f} ± {s['std']:.4f} {s['mean'] - base['mean']:+9.4f} {d_eff:22.3f}")
    print(f"seed std of the unique-data arm: {base['std']:.4f} nats")
    ladder.save_json(out, out_dir(a.variant) / "repeat.json")


def cmd_published(lab, a, v):
    c = laws.CHINCHILLA
    print("Compute-optimal allocation under Hoffmann et al.'s approach-3 fit (your compute_optimal), C = 6ND:")
    for C in (5.76e23, 3.8e25):
        N, D = lab.compute_optimal(C, c)
        Nb, Db = lab.compute_optimal(C, laws.BESIROGLU)
        print(f"  C = {C:.3g}: N_opt = {N / 1e9:.1f}B, D_opt = {D / 1e12:.2f}T ({D / N:.0f} tokens/param); "
              f"Besiroglu refit: N_opt = {Nb / 1e9:.1f}B, D_opt = {Db / 1e12:.2f}T ({Db / Nb:.0f} tokens/param)")
    print("\nReleased small models, priced with the same law (an extrapolation: the law was fitted on other data):")
    models = [("Llama 3 8B, ~15T tokens (Meta blog)", 8.0e9, 15e12), ("Gemma 3 1B, 2T tokens (report 2.2)", 1.0e9, 2e12),
              ("Qwen3-0.6B, ~36T tokens (report 3.1)", 0.6e9, 36e12), ("Chinchilla 70B, 1.4T tokens", 70e9, 1.4e12)]
    print(f"  {'model':38s} {'tok/param':>9} | Hoffmann fit: loss, extra compute | Besiroglu refit: loss, extra compute")
    for name, N, D in models:
        o, ob = lab.overtraining_overhead(N, D, c), lab.overtraining_overhead(N, D, laws.BESIROGLU)
        print(f"  {name:38s} {D / N:9.0f} | {float(lab.chinchilla_loss(N, D, c)):.3f}, {o:+8.0%} | "
              f"{float(lab.chinchilla_loss(N, D, laws.BESIROGLU)):.3f}, {ob:+8.0%}")
    L = float(laws.optimal_loss(1e23))
    print(f"\nInference-aware sizing (Sardana et al.) for the compute-optimal loss at 1e23 FLOPs ({L:.3f}):")
    for Dinf in (0.0, 1e12, 1e13, 1e14):
        r = laws.inference_aware(L, Dinf)
        print(f"  inference tokens {Dinf:7.0e}: N = {r['N'] / 1e9:6.2f}B, D = {r['D'] / 1e12:6.2f}T ({r['tokens_per_param']:6.0f} tokens/param)")
    print("\nRepetition (Muennighoff et al., your effective_data): worth of D tokens drawn from U unique tokens")
    for ep in (1, 2, 4, 8, 16, 40, 100):
        print(f"  {ep:4d} epochs: D' / D = {float(lab.effective_data(1.0, float(ep))) / ep:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["published", "isoflop", "repeat"])
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--print", action="store_true", help="print the training commands instead of running them")
    ap.add_argument("--lab", default=None, help="lab (default) or solution")
    a = ap.parse_args(argv)
    lab = common.lab_module(a.lab, HERE)
    v = VARIANTS[a.variant]
    {"published": cmd_published, "isoflop": cmd_isoflop, "repeat": cmd_repeat}[a.cmd](lab, a, v)


if __name__ == "__main__":
    main()
