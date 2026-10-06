"""Lab 11.2: predict a downstream metric of the 11.1 ladder from its loss, and fit an observational law on published scores.

    python labs/module-11/lesson-02/downstream_lab.py ladder              # needs the 11.1 iso-FLOP runs; about 5 min CPU
    python labs/module-11/lesson-02/downstream_lab.py observational       # downloads one 28 kB CSV at a pinned commit

``ladder``: every finished 11.1 run is scored on a 4-way cloze task built from the *test* split (500 items, context 48
tokens, continuation 8): accuracy, probability on the correct option, NLL of the correct option, and greedy exact match
of 4 tokens. Then the two-step prediction (Llama 3 section 3.2.1; OLMo ladder, arXiv 2412.04403): step 1, task NLL as
L(N, D) fitted on the two smaller budgets; step 2, accuracy as a sigmoid of task NLL (your TODOs) fitted on the same runs;
both checked on the held-out largest budget, against a direct fit of accuracy on log-compute.

``observational``: the benchmark table of Ruan et al.'s observational scaling laws (arXiv 2405.10938), file
``eval_results/base_llm_benchmark_eval.csv`` of github.com/ryoungj/ObsScaling at commit 4d6e1e4 (Apache-2.0), checked by
SHA-256. PCA of the benchmark scores (your ``explained_variance``); then a held-out test: fit GSM8K as a sigmoid of the
top principal components on models below a FLOPs cutoff, predict the models above it, and compare with a fit on
log-FLOPs alone.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

from frontierlab.data.loader import TokenData  # noqa: E402
from frontierlab.model import LM, ModelConfig  # noqa: E402
from frontierlab.scaling import downstream as ds  # noqa: E402
from frontierlab.scaling import fit as sfit  # noqa: E402
from frontierlab.stats import bootstrap_ci  # noqa: E402

OBS_URL = ("https://raw.githubusercontent.com/ryoungj/ObsScaling/4d6e1e43fd2635d04654aa77d1df9d5266ea0382/"
           "eval_results/base_llm_benchmark_eval.csv")
OBS_SHA256 = "511996815735a4c46251dc585dcc525e370d148201f86561e6f927f9df2d0db8"
BENCH = ["MMLU", "ARC-C", "HellaSwag", "Winograd", "TruthfulQA", "GSM8K"]


def load_model(run_dir: Path) -> LM:
    ck = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    m = LM(ModelConfig(**ck["config"]))
    m.load_state_dict(ck["model"])
    return m.eval()


def score_ladder(variant: str, n_items: int, device: str) -> list[dict]:
    from importlib import util
    spec = util.spec_from_file_location("ladder_lab", HERE.parent / "lesson-01" / "ladder_lab.py")
    ll = util.module_from_spec(spec)
    spec.loader.exec_module(ll)
    v = ll.VARIANTS[variant]
    res_path = ll.out_dir(variant) / "results.json"
    if not res_path.exists():
        raise SystemExit(f"{res_path} missing: run `ladder_lab.py isoflop --variant {variant}` first (lesson 11.1)")
    runs = json.loads(res_path.read_text())["runs"]
    cache = ll.out_dir(variant) / f"downstream_{n_items}.json"
    done = {r["run"]: r for r in json.loads(cache.read_text())} if cache.exists() else {}
    items = ds.build_cloze(TokenData("test", v["data"]), n_items=n_items, ctx=48, cont=8, seed=0)
    out = []
    for r in runs:
        if r["run"] not in done:
            m = load_model(ll.out_dir(variant) / r["run"]).to(device)
            sc = ds.score_cloze(m, items, device=device)
            em = ds.exact_match(m, items, k=4, device=device)
            done[r["run"]] = {"run": r["run"], **{k: sc[k] for k in ("acc", "p_correct", "brier", "nll_correct")},
                              "acc_items": sc["items"]["acc"], **em}
            cache.write_text(json.dumps(list(done.values())))
            print(f"  scored {r['run']}: acc {sc['acc']:.3f}", flush=True)
        out.append({**r, **done[r["run"]]})
    return out


def cmd_ladder(lab, a):
    rows = score_ladder(a.variant, a.items, a.device)
    print(f"\nCloze task on the test split ({a.items} items, 4 options, chance 0.25), all 11.1 runs:")
    print(f"{'run':>16} {'C':>8} {'val loss':>8} {'task NLL':>8} {'p_corr':>7} {'acc':>6} {'acc 95% CI':>14} {'tok acc':>7} {'EM@4':>6} {'tok^4':>6}")
    for r in sorted(rows, key=lambda r: r["C"]):
        _, lo, hi = bootstrap_ci(r["acc_items"], n_boot=2000)
        print(f"{r['run']:>16} {r['C']:8.2g} {r['loss']:8.4f} {r['nll_correct']:8.4f} {r['p_correct']:7.3f} {r['acc']:6.3f} "
              f"[{lo:.3f}, {hi:.3f}] {r['token_acc']:7.3f} {r['exact_match']:6.3f} {float(lab.exact_match_from_token_acc(r['token_acc'], 4)):6.3f}")
    budgets = sorted({r["budget"] for r in rows})
    tr = [r for r in rows if r["budget"] < budgets[-1]]
    te = [r for r in rows if r["budget"] == budgets[-1]]
    N = lambda rs: np.array([r["N_total"] for r in rs], float)
    D = lambda rs: np.array([r["D"] for r in rs], float)
    nll_fit = sfit.fit_parametric(N(tr), D(tr), [r["nll_correct"] for r in tr])
    sig = ds.fit_sigmoid([r["nll_correct"] for r in tr], [r["acc"] for r in tr], lo=0.25, hi=1.0)
    direct = np.polyfit(np.log10([r["C"] for r in tr]), [r["acc"] for r in tr], 1)
    print(f"\nStep 1 (task NLL = E + A N^-a + B D^-b on {len(tr)} runs): E = {nll_fit['E']:.3f}, alpha = {nll_fit['alpha']:.2f}, beta = {nll_fit['beta']:.2f}")
    print(f"Step 2 (acc = 0.25 + 0.75 / (1 + exp(s (NLL - x0)))): x0 = {sig['x0']:.3f}, s = {sig['s']:.2f}, RMS {sig['rms']:.4f}")
    print(f"\nHeld-out budget {budgets[-1]:.0e}:")
    print(f"{'run':>16} {'acc':>6} {'two-step':>8} {'step 2 on measured NLL':>22} {'direct':>7} | {'task NLL':>8} {'pred':>7}")
    errs = {"two-step": [], "step 2 on measured NLL": [], "direct": []}
    for r in te:
        nll_hat = float(sfit.predict(nll_fit, r["N_total"], r["D"]))
        two = lab.two_step(nll_hat, sig)
        orc = lab.two_step(r["nll_correct"], sig)                     # step 2 alone, with the measured NLL
        dirc = float(np.polyval(direct, math.log10(r["C"])))
        errs["two-step"].append(two - r["acc"]); errs["step 2 on measured NLL"].append(orc - r["acc"]); errs["direct"].append(dirc - r["acc"])
        print(f"{r['run']:>16} {r['acc']:6.3f} {two:8.3f} {orc:22.3f} {dirc:7.3f} | {r['nll_correct']:8.4f} {nll_hat:7.4f}")
    for k, e in errs.items():
        print(f"  mean |error|, {k}: {np.mean(np.abs(e)):.3f}")
    common.ladder.save_json({"rows": [{k: v for k, v in r.items() if k != "acc_items"} for r in rows], "nll_fit": nll_fit,
                             "sigmoid": sig, "errors": errs}, common.RUNS / "l112" / a.variant / "results.json")


def fetch_obs(path: Path) -> bytes:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {OBS_URL}")
        with urllib.request.urlopen(OBS_URL, timeout=60) as f:
            path.write_bytes(f.read())
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != OBS_SHA256:
        raise SystemExit(f"{path}: SHA-256 mismatch (expected {OBS_SHA256}); delete it and retry")
    return data


def cmd_observational(lab, a):
    data = fetch_obs(common.RUNS / "l112" / "base_llm_benchmark_eval.csv")
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    num = lambda s: float(s) if s not in ("", None) else math.nan
    X = np.array([[num(r[b]) for b in BENCH] for r in rows])
    flops = np.array([num(r["FLOPs (1E21)"]) * 1e21 for r in rows])
    keep = ~np.isnan(X).any(1)
    print(f"{len(rows)} models in the table; {keep.sum()} have all of {', '.join(BENCH)}")
    ev = lab.explained_variance(X[keep])
    print("variance explained by PC1..PC4: " + ", ".join(f"{x:.3f}" for x in ev[:4]) + f"  (top 3: {ev[:3].sum():.3f})")
    p = ds.pca_capabilities(X[keep], 3)
    print("PC1 loadings: " + ", ".join(f"{b} {w:+.2f}" for b, w in zip(BENCH, p["loadings"][0])))
    Xk, fk = X[keep], flops[keep]
    names = [r["Model"] for r, k in zip(rows, keep) if k]
    has_f = ~np.isnan(fk)
    cut = a.cutoff
    tr, te = has_f & (fk < cut), has_f & (fk >= cut)
    print(f"\nHold-out by compute: fit on the {int(tr.sum())} models below {cut:.0e} FLOPs, predict the {int(te.sum())} "
          "at or above it. Mean |error| (accuracy, 0-1):")
    lf = np.log10(np.where(has_f, fk, 1.0))[:, None] - 22
    print(f"  {'target':>8} {'2 PCs of the other 5 benchmarks':>32} {'log-FLOPs alone':>16}")
    out = {}
    for target in ("MMLU", "ARC-C", "GSM8K"):
        g = BENCH.index(target)
        others = [i for i in range(len(BENCH)) if i != g]               # the target is never its own predictor
        mu, sd = Xk[tr][:, others].mean(0), Xk[tr][:, others].std(0)   # standardised on the training models only
        Z = (Xk[:, others] - mu) / sd
        _, _, Vt = np.linalg.svd(Z[tr], full_matrices=False)
        S = Z @ Vt[:2].T
        fit_pc, fit_c = ds.fit_logistic(S[tr], Xk[tr, g]), ds.fit_logistic(lf[tr], Xk[tr, g])
        pred = ds.predict_logistic(fit_pc, S[te])
        e_pc = float(np.abs(pred - Xk[te, g]).mean())
        e_c = float(np.abs(ds.predict_logistic(fit_c, lf[te]) - Xk[te, g]).mean())
        out[target] = (e_pc, e_c)
        print(f"  {target:>8} {e_pc:32.3f} {e_c:16.3f}")
        if target == "GSM8K":
            tn, y = [n for n, t in zip(names, te) if t], Xk[te, g]
            print("\n  GSM8K, the five largest held-out errors of the PC fit:")
            for i in np.argsort(-np.abs(pred - y))[:5]:
                print(f"    {tn[i][:42]:42s} measured {y[i]:.3f}  predicted {pred[i]:.3f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["ladder", "observational"])
    ap.add_argument("--variant", default="cpu")
    ap.add_argument("--items", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cutoff", type=float, default=1e23, help="FLOPs cutoff for the observational hold-out")
    ap.add_argument("--lab", default=None)
    a = ap.parse_args(argv)
    lab = common.lab_module(a.lab, HERE)
    {"ladder": cmd_ladder, "observational": cmd_observational}[a.cmd](lab, a)


if __name__ == "__main__":
    main()
