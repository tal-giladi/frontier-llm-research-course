"""Lab 10.2: train a quality classifier, filter a web pool with it, and judge it by a matched training ablation.

    python -m frontierlab.datax.sources prepare annot --docs 24000     # 45 MB, ~1 min (Llama-3-70B labels)
    python -m frontierlab.datax.sources prepare web --docs 20000       # 38 MB, ~2 min (FineWeb, unfiltered)
    python -m frontierlab.datax.sources prepare wiki --docs 4000       # 36 MB, ~1 min (neutral held-out set)
    python labs/module-10/lesson-02/filter_ablation.py                 # free CPU: ~45 min

Steps (each cached, so a rerun continues where it stopped):

1. **Classifier.** Fit ``frontierlab.datax.quality`` (hashed uni+bigrams, linear regression on the 0–5
   score) on 80% of the annotated documents (split by SHA-1 bucket), report RMSE, Spearman and F1 at
   score 3 on the other 20%, against a majority-class baseline. This is the number papers report and the
   one the lab does *not* decide on.
2. **Filter.** Score the first ``--pool`` training documents of the web source; keep the top 30%, 10% and
   3% (your ``keep_top``). The unfiltered arm is the whole pool.
3. **Ablation.** Train every arm at the same token budget (``--steps × --batch × --seq``) with 3 seeds;
   the filtered arms repeat their data (your ``epochs_needed``; the exact epochs are in each run's
   ``mixture_accounting.json``).
4. **Score** on Data-v0 validation (FineWeb-Edu: the classifier's target distribution), the web pool's
   validation split (the unfiltered distribution), Wikipedia validation (neither), and LAMBADA; compare
   each filtered arm with the unfiltered one by seed, and apply the pre-stated rule (your ``decide``) to
   the primary metric, Wikipedia held-out loss.
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
from frontierlab.datax import arms, quality
from frontierlab.datax.mixture import MixtureSpec, SourceRef, make_source_subset
from frontierlab.datax.neardup import decode_docs
from frontierlab.datax.sources import M10_ROOT, load_tokenizer
from frontierlab.labkit import load_path

HERE = Path(__file__).resolve().parent
VARIANTS = {
    "cpu": dict(preset="toy", steps=500, batch=16, seq=128, lr=3e-3, warmup=50, seeds=[0, 1, 2], device="cpu",
                windows=256, lambada=1000, pool=8000),
    "t4": dict(preset="pilot-10m", steps=3000, batch=32, seq=512, lr=3e-3, warmup=150, seeds=[0, 1, 2], device="cuda",
               windows=256, lambada=5153, pool=200000),
    "main": dict(preset="pilot-30m", steps=6000, batch=64, seq=1024, lr=3e-3, warmup=300, seeds=[0, 1, 2], device="cuda",
                 windows=256, lambada=5153, pool=2000000),
}
FRACS = {"top30": 0.30, "top10": 0.10, "top03": 0.03}


def fit_classifier(lab, out: Path, annot: Path, tok) -> dict:
    path = out / "classifier.json"
    if path.exists():
        return json.loads(path.read_text())
    t0 = time.time()
    d = TokenData("train", annot)
    texts = decode_docs(d, tok)
    rows = [json.loads(x) for x in (annot / "train_provenance.jsonl").read_text(encoding="utf-8").splitlines()]
    y = np.array([float(r["score"]) for r in rows])
    fit_mask = np.array([int(r["sha1"][:8], 16) % 10 < 8 for r in rows])
    feats = [lab.bucket_features(t) for t in texts]
    t1 = time.time()
    model = quality.fit(None, y[fit_mask], feats=[f for f, k in zip(feats, fit_mask) if k], epochs=6, lr=0.5)
    pred = quality.predict(model, feats=[f for f, k in zip(feats, fit_mask) if not k])
    yv = y[~fit_mask]
    major = float(np.bincount((y[fit_mask] >= 3).astype(int)).argmax())
    res = {"n_fit": int(fit_mask.sum()), "n_val": int((~fit_mask).sum()),
           "score_histogram": np.bincount(y.astype(int), minlength=6).tolist(),
           "val_rmse": float(np.sqrt(np.mean((pred - yv) ** 2))), "val_rmse_predict_mean": float(np.std(yv)),
           "val_spearman": quality.spearman(pred, yv), "val_f1_at_3": lab.f1_at(pred, yv, 3.0),
           # the same F1 with the threshold moved so that the predicted positive rate equals the true one
           "val_f1_at_matched_rate": quality.binary_metrics(pred >= np.quantile(pred, 1 - (yv >= 3).mean()), yv >= 3)["f1"],
           "val_positive_rate": float((yv >= 3).mean()), "majority_class_is_positive": bool(major),
           "metrics_at_3": quality.binary_metrics(pred >= 3, yv >= 3),
           "seconds_features": round(t1 - t0, 1), "seconds_fit": round(time.time() - t1, 1)}
    import torch
    torch.save(model.state_dict(), out / "classifier.pt")
    path.write_text(json.dumps(res, indent=2))
    return res


def build_arms(lab, out: Path, web: Path, pool: int, tok) -> dict:
    path = out / "arms.json"
    if path.exists():
        return json.loads(path.read_text())
    import torch
    model = quality.BagClassifier()
    model.load_state_dict(torch.load(out / "classifier.pt"))
    d = TokenData("train", web)
    texts = decode_docs(d, tok, pool)
    scores = quality.predict(model, feats=[lab.bucket_features(t) for t in texts])
    np.save(out / "pool_scores.npy", scores)
    make_source_subset(web, out / "src-unfiltered", np.arange(len(texts)), note=f"first {len(texts)} web train documents")
    roots = {"unfiltered": str(out / "src-unfiltered")}
    for name, frac in FRACS.items():
        keep = lab.keep_top(scores, frac)
        make_source_subset(web, out / f"src-{name}", keep, note=f"classifier top {frac:.0%} of the pool")
        roots[name] = str(out / f"src-{name}")
    info = {"roots": roots, "pool_docs": len(texts), "score_quantiles": np.quantile(scores, [0.1, 0.5, 0.9, 0.97]).tolist(),
            "tokens": {k: int(TokenData("train", v).tokens.size) for k, v in roots.items()}}
    path.write_text(json.dumps(info, indent=2))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--out", type=Path, default=Path("runs/m10/l102"))
    ap.add_argument("--annot", type=Path, default=M10_ROOT / "annot")
    ap.add_argument("--web", type=Path, default=M10_ROOT / "web")
    ap.add_argument("--wiki", type=Path, default=M10_ROOT / "wiki")
    ap.add_argument("--classifier-only", action="store_true")
    a = ap.parse_args(argv)
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    v = VARIANTS[a.variant]
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    a.out.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer()
    t0 = time.time()
    clf = fit_classifier(lab, a.out, a.annot, tok)
    print("classifier (validation 20% of the annotations):", json.dumps({k: clf[k] for k in (
        "n_fit", "n_val", "score_histogram", "val_rmse", "val_rmse_predict_mean", "val_spearman", "val_f1_at_3",
        "val_f1_at_matched_rate", "val_positive_rate", "seconds_features", "seconds_fit")}))
    if a.classifier_only:
        return
    info = build_arms(lab, a.out, a.web, v["pool"], tok)
    budget = v["steps"] * v["batch"] * v["seq"]
    print(f"pool {info['pool_docs']} documents; tokens per arm source: {info['tokens']}; budget {budget:,} tokens")
    for k, n in info["tokens"].items():
        print(f"  {k:10s} epochs needed: {lab.epochs_needed(budget, n):.2f}")
    sets = {"edu": str(DEFAULT_OUT), "web": str(a.web), "wiki": str(a.wiki)}
    results = {}
    for name, root in info["roots"].items():
        spec = MixtureSpec([SourceRef(name, root, 1.0)], name=f"l102-{name}")
        results[name] = {}
        for s in v["seeds"]:
            run = arms.train_arm(a.out / f"{name}-s{s}", spec, preset=v["preset"], steps=v["steps"], batch=v["batch"],
                                 seq=v["seq"], lr=v["lr"], warmup=v["warmup"], seed=s, device=v["device"],
                                 question="does classifier filtering improve held-out loss at equal tokens? (10.2)")
            results[name][s] = arms.score(run, sets, windows=v["windows"], seq=v["seq"], n_lambada=v["lambada"],
                                          device=v["device"])
    rows = arms.table(results, "unfiltered", ["wiki", "edu", "web", "lambada"], v["seeds"], decide=lab.decide)
    print("\nseed-level comparison with the unfiltered pool (arm − unfiltered; lambada: log-prob, higher is better)")
    arms.print_table(rows)
    for m in ("wiki", "edu", "web", "lambada"):
        print(f"seed std of the unfiltered arm on {m}: {arms.seed_std(results, 'unfiltered', m, v['seeds']):.4f}")
    (a.out / "summary.json").write_text(json.dumps({"classifier": clf, "arms": info, "rows": rows}, indent=2))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
