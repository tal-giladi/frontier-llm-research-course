"""Lab 10.4: RegMix over many tiny runs, a confirmation run, and micro-anneals that score candidate datasets.

    python labs/module-10/lesson-04/mixture_lab.py regmix        # 24 tiny runs + fits + 9 confirmation runs, ~45 min CPU
    python labs/module-10/lesson-04/mixture_lab.py anneal        # 1 stable run + 9 micro-anneals, ~15 min CPU

Sources: Data-v0 (``edu``), FineWeb (``web``), Wikipedia (``wiki``), FineMath (``math``), all prepared with
``python -m frontierlab.datax.sources prepare ...`` (lesson 10.1). Target of the regression: the mean of the four
sources' validation losses ("avg": the data mixing law's Eq. 8 with equal domain weights s_i = 1/4).

``regmix``: 24 mixtures sampled RegMix's way around the sources' token shares (your ``sample_mixtures``), one tiny
run each; ridge (yours), ridge with pairwise terms and the mixing law (``frontierlab.datax.regmix``) fitted to
the targets; leave-one-out rank correlation (Spearman), the paper's metric; the predicted-best mixture
(average of the 100 best of 100,000 sampled); then a confirmation at the same budget with 3 seeds: predicted
best vs uniform vs natural (token-proportional), paired by seed.

``anneal``: a stable-phase run on edu + web (constant learning rate), then micro-anneals of ``--anneal-steps``
from its weights with the learning rate decayed linearly to zero (your ``anneal_lr`` is the schedule of
``frontierlab.datax.train --anneal``): control (edu + web only), + wiki 50/50, + math 50/50, 3 seeds each.
Each candidate is scored on its target domain and on Data-v0 (the guard) with your ``anneal_decision``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from frontierlab.data.loader import TokenData
from frontierlab.data.prepare import DEFAULT_OUT
from frontierlab.datax import anneal, arms, regmix
from frontierlab.datax import evaluate as ev
from frontierlab.datax.mixture import MixtureSpec, SourceRef
from frontierlab.datax.sources import M10_ROOT
from frontierlab.labkit import load_path

HERE = Path(__file__).resolve().parent
NAMES = ["edu", "web", "wiki", "math"]
VARIANTS = {
    "cpu": dict(preset="toy", steps=200, batch=16, seq=128, lr=3e-3, warmup=20, n_runs=24, seeds=[0, 1, 2],
                device="cpu", windows=128, stable_steps=400, anneal_steps=100),
    "t4": dict(preset="pilot-10m", steps=1000, batch=32, seq=512, lr=3e-3, warmup=50, n_runs=32, seeds=[0, 1, 2],
               device="cuda", windows=256, stable_steps=3000, anneal_steps=500),
    "main": dict(preset="pilot-10m", steps=2000, batch=64, seq=1024, lr=3e-3, warmup=100, n_runs=64, seeds=[0, 1, 2],
                 device="cuda", windows=256, stable_steps=8000, anneal_steps=1000),
}


def roots() -> dict:
    return {"edu": str(DEFAULT_OUT), "web": str(M10_ROOT / "web"), "wiki": str(M10_ROOT / "wiki"),
            "math": str(M10_ROOT / "math")}


def spec_of(w, name: str) -> MixtureSpec:
    r = roots()
    return MixtureSpec([SourceRef(n, r[n], float(x)) for n, x in zip(NAMES, w) if x > 0], name=name)


def regmix_cmd(lab, a, v):
    r = roots()
    prior = np.array([TokenData("train", r[n]).tokens.size for n in NAMES], dtype=np.float64)
    X = lab.sample_mixtures(v["n_runs"], prior, seed=0)
    # RegMix's block of 100 windows rounds weights to 1%; use the realised weights as the regression inputs
    from frontierlab.datax.mixture import window_counts
    Xr = np.stack([window_counts(x, 100) / 100 for x in X])
    sets = {n: r[n] for n in NAMES}
    y, res = [], []
    t0 = time.time()
    for i, w in enumerate(Xr):
        run = arms.train_arm(a.out / f"rm-{i:02d}", spec_of(w, f"regmix-{i:02d}"), preset=v["preset"], steps=v["steps"],
                             batch=v["batch"], seq=v["seq"], lr=v["lr"], warmup=v["warmup"], seed=0, device=v["device"],
                             question="RegMix tiny run (10.4)")
        s = arms.score(run, sets, windows=v["windows"], seq=v["seq"], n_lambada=0, device=v["device"])
        res.append(s)
        y.append(ev.mean_of(s, "avg"))
    y = np.array(y)
    per = {n: np.array([ev.mean_of(s, n) for s in res]) for n in NAMES}
    print(f"{len(y)} tiny runs in {time.time() - t0:.0f}s; target (avg of 4 val losses) range {y.min():.4f} - {y.max():.4f}")
    print("mixture (edu web wiki math) -> target, for the 5 best and 5 worst runs:")
    for i in list(np.argsort(y)[:5]) + list(np.argsort(y)[-5:]):
        print(f"  {np.round(Xr[i], 2)} -> {y[i]:.4f}")
    loo = {"ridge (yours)": lab.loo_ridge(Xr, y),
           "ridge2": regmix.loo_predictions("ridge2", Xr, y, l2=1e-3),
           "law (Eq. 8: sum of per-domain Eq. 7)": np.mean(
               [regmix.loo_predictions("law", Xr, per[n], steps=1500) for n in NAMES], axis=0)}
    for k, p in loo.items():
        print(f"leave-one-out {k:40s} Spearman {regmix.rank_corr(p, y):+.3f}  RMSE {np.sqrt(np.mean((p - y) ** 2)):.4f}")
    laws = {n: regmix.fit_mixing_law(Xr, per[n], steps=1500) for n in NAMES}
    for n, f in laws.items():
        print(f"  law for {n:5s} val loss: c={f['c']:.3f} k={f['k']:.3f} t={np.round(f['t'], 2)}")
    w = lab.ridge_fit(Xr, y)
    fit = {"kind": "ridge", "w": w, "quadratic": False}
    best = regmix.best_mixture(regmix.fit_ridge(Xr, y, quadratic=True, l2=1e-3), 4, prior=prior)
    best_lin = regmix.best_mixture(fit, 4, prior=prior)
    print(f"predicted best (ridge2, top-100 average): {np.round(best['mixture'], 3)} -> {best['predicted']:.4f}")
    print(f"predicted best (ridge, top-100 average):  {np.round(best_lin['mixture'], 3)} -> {best_lin['predicted']:.4f}")
    # confirmation
    cands = {"best": best["mixture"], "uniform": np.ones(4) / 4, "natural": prior / prior.sum()}
    conf = {}
    for name, wv in cands.items():
        conf[name] = {}
        for s in v["seeds"]:
            run = arms.train_arm(a.out / f"conf-{name}-s{s}", spec_of(wv, f"conf-{name}"), preset=v["preset"],
                                 steps=v["steps"], batch=v["batch"], seq=v["seq"], lr=v["lr"], warmup=v["warmup"], seed=s,
                                 device=v["device"], question="RegMix confirmation (10.4)")
            conf[name][s] = arms.score(run, sets, windows=v["windows"], seq=v["seq"], n_lambada=0, device=v["device"])
    print("\nconfirmation, 3 seeds (arm − uniform, target avg and each domain):")
    rows = arms.table(conf, "uniform", ["avg", *NAMES], v["seeds"])
    arms.print_table(rows)
    pred = {k: float(regmix.predict(regmix.fit_ridge(Xr, y, quadratic=True, l2=1e-3), np.asarray(x)[None])[0]) for k, x in cands.items()}
    print("predicted (ridge2) vs measured mean target:", {k: (round(pred[k], 4), round(float(np.mean([ev.mean_of(conf[k][s], 'avg') for s in v['seeds']])), 4)) for k in cands})
    print(f"seed std of the uniform arm on avg: {arms.seed_std(conf, 'uniform', 'avg', v['seeds']):.4f}")
    (a.out / "regmix.json").write_text(json.dumps({"X": Xr.tolist(), "y": y.tolist(), "per": {k: x.tolist() for k, x in per.items()},
                                                   "best": best["mixture"].tolist(), "rows": rows}, indent=2))


def anneal_cmd(lab, a, v):
    r = roots()
    base = MixtureSpec([SourceRef("edu", r["edu"], 0.5), SourceRef("web", r["web"], 0.5)], name="stable")
    stable = arms.train_arm(a.out / "stable", base, preset=v["preset"], steps=v["stable_steps"], batch=v["batch"],
                            seq=v["seq"], lr=v["lr"], warmup=v["warmup"], seed=0, device=v["device"],
                            extra=["--schedule", "constant"], question="stable phase for micro-anneals (10.4)")
    cands = {"wiki": SourceRef("wiki", r["wiki"], 1.0), "math": SourceRef("math", r["math"], 1.0)}
    runs = anneal.run_microanneals(a.out, stable / "checkpoint.pt", base, cands, preset=v["preset"],
                                   steps=v["anneal_steps"], batch=v["batch"], seq=v["seq"], lr=v["lr"],
                                   warmup=max(1, v["anneal_steps"] // 10), seeds=v["seeds"], device=v["device"])
    sets = {n: r[n] for n in NAMES}
    res = {k: {s: arms.score(p, sets, windows=v["windows"], seq=v["seq"], n_lambada=0, device=v["device"])
               for s, p in d.items()} for k, d in runs.items()}
    st = arms.score(stable, sets, windows=v["windows"], seq=v["seq"], n_lambada=0, device=v["device"])
    print("stable checkpoint:", {n: round(ev.mean_of(st, n), 4) for n in NAMES})
    rows = arms.table(res, "control", NAMES, v["seeds"])
    arms.print_table(rows)
    for cand in cands:
        t = ev.seed_level([ev.mean_of(res["control"][s], cand) for s in v["seeds"]],
                          [ev.mean_of(res[cand][s], cand) for s in v["seeds"]])
        g = ev.seed_level([ev.mean_of(res["control"][s], "edu") for s in v["seeds"]],
                          [ev.mean_of(res[cand][s], "edu") for s in v["seeds"]])
        print(f"{cand}: target {t['mean_diff']:+.4f} [{t['ci'][0]:+.4f}, {t['ci'][1]:+.4f}], guard (edu) "
              f"{g['mean_diff']:+.4f} [{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}] -> {lab.anneal_decision(t['ci'], g['ci'], 0.02)}")
    (a.out / "anneal.json").write_text(json.dumps({"rows": rows}, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["regmix", "anneal"])
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--out", type=Path, default=Path("runs/m10/l104"))
    a = ap.parse_args(argv)
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    v = VARIANTS[a.variant]
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    t0 = time.time()
    (regmix_cmd if a.cmd == "regmix" else anneal_cmd)(lab, a, v)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
