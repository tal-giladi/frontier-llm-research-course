"""Lab 18.3: time horizons from METR's released runs, a planted-contamination experiment, Eval Suite v3.

    python labs/module-18/lesson-03/eval_lab.py                       # free CPU, all parts (about 3-6 minutes)
    python labs/module-18/lesson-03/eval_lab.py --part horizon        # part A only (downloads 15 MB once)
    python labs/module-18/lesson-03/eval_lab.py --part contamination  # part B only
    python labs/module-18/lesson-03/eval_lab.py --variant main --print

Part A (analysis of released data): METR's Time Horizon 1.1 runs (pinned commit, SHA-256 checked; METR's
repository states no licence, so the file is downloaded to runs/m18/data and never committed). Your
``horizon_from_fit`` turns each model's logistic fit into its 50% and 80% horizons; the fit is compared with METR's
published numbers; a hierarchical bootstrap (families, tasks, runs) gives the interval; three sensitivity checks
show how much the number depends on choices: task weighting, dropping one task source, and the 80% horizon.

Part B (a measured experiment): your toy model from Module 13 (``runs/m13/project/toy-s0/s3-distill``; the Module 12
warm start if it is missing), a copy fine-tuned on 100 of Eval v3's 200 published items (a planted leak, mixed
with training-split replay), and the control: the same fine-tune with replay in place of the leak. Eval v3 is run on both, once with the leak declared as training data and once
undeclared. Your ``palm_contaminated`` and ``min_k_percent`` are checked against the suite's on every item.

Part C: the lifecycle cards of the benchmarks a frontier report would cite, and your ``v3_verdict``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch

from frontierlab.evals.suite_v3 import contamination as C
from frontierlab.evals.suite_v3 import horizon as Hz
from frontierlab.evals.suite_v3 import lifecycle as L
from frontierlab.evals.suite_v3 import toy as T3
from frontierlab.evals.suite_v3.core import compare, report
from frontierlab.labkit import load_path
from frontierlab.posttrain.sft import ensure_sft, load_policy, save_policy, sft_loss
from frontierlab.posttrain.tasks import TAGS, encode_sft, make_problems, problems_from, split_problems

HERE = Path(__file__).resolve().parent
ROOT = Path("runs/m18/l183")
DATA = Path("runs/m18/data")
LEARNER = Path("runs/m13/project/toy-s0/s3-distill/policy.pt")
MODELS = ["GPT-4 0314", "Claude 3.7 Sonnet (Inspect)", "o3 (Inspect)", "GPT-5 (Inspect)", "Claude Opus 4.5 (Inspect)",
          "GPT-5.2", "Claude Opus 4.6 (Inspect)"]


def learner_model() -> tuple[Path, str]:
    if LEARNER.exists():
        return LEARNER, "Module 13 project, toy pipeline seed 0, stage S3 (distillation)"
    return ensure_sft("runs/m12/sft"), "Module 12 warm start (the Module 13 project checkpoint was not found)"


# --------------------------------------------------------------------------- part A

def part_horizon(lab, runs_path: Path | None, n_boot: int):
    path = runs_path or Hz.download(DATA / "metr_th11_runs.jsonl")
    dates = Hz.load_dates(Hz.download_dates(DATA / "metr_release_dates.yaml"))
    t0 = time.time()
    print(f"(A) METR runs: {path} ({Hz.METR_RUNS['repo']} @ {Hz.METR_RUNS['commit'][:7]})")
    print(f"  {'model':30s} {'runs':>5s} {'tasks':>5s} {'h50 (min)':>10s} {'METR':>7s} {'h80':>7s} {'beta':>5s}  "
          f"your h50 check")
    fits = {}
    for m in MODELS:
        rows = Hz.load_runs(path, m)
        f = Hz.fit_horizon(rows)
        mine = lab.horizon_from_fit(f["a"], -f["beta"], 0.5)
        fits[m] = f
        pub = Hz.METR_PUBLISHED.get(m)
        print(f"  {m:30s} {f['n_runs']:5d} {f['n_tasks']:5d} {f['h50']:10.1f} {pub if pub else '-':>7} "
              f"{lab.horizon_from_fit(f['a'], -f['beta'], 0.8):7.1f} {f['beta']:5.2f}  "
              f"{'ok' if math.isclose(mine, f['h50'], rel_tol=1e-9) else 'MISMATCH'}")
    for m in ("Claude 3.7 Sonnet (Inspect)", "GPT-5.2"):
        ci = Hz.hierarchical_bootstrap(Hz.load_runs(path, m), n_boot=n_boot)
        print(f"  {m}: h50 95% interval [{ci['h50_ci'][0]:.0f}, {ci['h50_ci'][1]:.0f}] min, h80 "
              f"[{ci['h80_ci'][0]:.0f}, {ci['h80_ci'][1]:.0f}] ({n_boot} hierarchical resamples)")
    print("  sensitivity, o3 (Inspect) and GPT-5.2:")
    for m in ("o3 (Inspect)", "GPT-5.2"):
        rows = Hz.load_runs(path, m)
        alt = {w: Hz.fit_horizon(rows, w)["h50"] for w in ("invsqrt_task_weight", "equal_task_weight", None)}
        no_swaa = Hz.fit_horizon([r for r in rows if r["task_source"] != "SWAA"])["h50"]
        no_re = Hz.fit_horizon([r for r in rows if r["task_source"] != "RE-Bench"])["h50"]
        print(f"    {m}: weights invsqrt {alt['invsqrt_task_weight']:.0f}, equal {alt['equal_task_weight']:.0f}, "
              f"none {alt[None]:.0f} min; without SWAA (the shortest tasks) {no_swaa:.0f}; without RE-Bench "
              f"{no_re:.0f}")
    xs, ys, names = [], [], []
    for m, f in fits.items():
        d = dates.get(m) or dates.get(m.replace(" (Inspect)", ""))
        if d:
            xs.append((date.fromisoformat(d) - date(2023, 1, 1)).days)
            ys.append(f["h50"])
            names.append(m)
    dt = Hz.doubling_time_days(xs, ys)
    print(f"  doubling time over these {len(xs)} models (least squares on log2 h50 vs release date): {dt:.0f} days. "
          f"METR's own 2023+ fit: 128.7 days [104, 158]; with 7 models and no interval this is a rough check only")
    print(f"  (A) {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- part B

def leak_model(base: Path, out: Path, n_leak: int = 100, steps: int = 150, seed: int = 0) -> tuple[Path, list[str]]:
    pub, _ = T3.v3_items()
    leak = pub[:n_leak]
    texts = [T3.item_text(p) for p in leak]
    path = out / "policy.pt"
    if path.exists():
        return path, texts
    torch.manual_seed(seed)
    model = load_policy(base)
    train_triples, held = split_problems(2)
    replay = make_problems(4000, 2, "+-", TAGS, seed=seed + 5, exclude=held)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)
    rng = random.Random(seed)
    for _ in range(steps):
        batch = (rng.sample(leak, 16) if leak else rng.sample(replay, 16)) + rng.sample(replay, 48)
        ids, mask = encode_sft(batch)
        loss = sft_loss(model, ids, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    save_policy(model, path, {"leak": n_leak, "steps": steps, "seed": seed, "base": str(base)})
    return path, texts


def part_contamination(lab):
    t0 = time.time()
    base_path, desc = learner_model()
    print(f"(B) learner model: {base_path} ({desc})")
    leaked_path, leak_texts = leak_model(base_path, ROOT / "leaked")
    control_path, _ = leak_model(base_path, ROOT / "replay-only", n_leak=0)
    train_triples, _ = split_problems(2)
    declared = [p.prompt + p.target for p in problems_from(train_triples, TAGS, 2, "+-")]
    clean, leaked = load_policy(base_path), load_policy(leaked_path)
    res = {"clean": T3.run_suite(clean, declared, benchmarks=()),
           "replay only (control)": T3.run_suite(load_policy(control_path), declared),
           "leaked, leak declared": T3.run_suite(leaked, declared + leak_texts),
           "leaked, leak undeclared": T3.run_suite(leaked, declared)}
    pub, _ = T3.v3_items()
    idx = C.build_index(declared + leak_texts, 6)
    mine = [lab.palm_contaminated(T3.item_text(p), idx, 6) for p in pub]
    ok_overlap = mine == res["leaked, leak declared"]["contamination"]["items"]["overlap"]
    lps = T3.item_logprobs(leaked, pub[:20])
    ok_mink = all(math.isclose(lab.min_k_percent(x), C.min_k_prob(x), rel_tol=1e-9) for x in lps)
    print(f"  your palm_contaminated agrees on all 200 items: {ok_overlap}; your min_k_percent agrees: {ok_mink}")
    print(f"  {'model':26s} {'add_greedy(v2)':>14s} {'published':>9s} {'fresh':>6s} {'gap [95% CI]':>22s} "
          f"{'flagged':>7s} {'AUC':>5s}  v3 check")
    for name, r in res.items():
        c = r["contamination"]
        g = c["fresh_gap"]
        print(f"  {name:26s} {r['summary']['add_greedy']:14.3f} {g['published']:9.3f} {g['fresh']:6.3f} "
              f"{g['gap']:+7.3f} [{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}] {c['flagged']:7d} {c['membership_auc']:5.2f}  "
              f"{'pass' if c['passes'] else 'FAIL'}")
    lk = res["leaked, leak declared"]["contamination"]
    auc_leaked = C.membership_auc(lk["items"]["min_k"][:100], lk["fresh"]["min_k"])
    auc_unleaked = C.membership_auc(lk["items"]["min_k"][100:], lk["fresh"]["min_k"])
    print(f"  membership AUC (Min-K% Prob, leaked model): the 100 leaked items vs fresh {auc_leaked:.2f}; the 100 "
          f"published items that were not leaked vs fresh {auc_unleaked:.2f} (0.5 = no signal)")
    ctl, lk2 = res["replay only (control)"]["contamination"], res["leaked, leak undeclared"]["contamination"]
    for name, key in (("published", "items"), ("fresh", "fresh")):
        m = np.mean(lk2[key]["correct"]) - np.mean(ctl[key]["correct"])
        print(f"  leak effect on {name:9s} items (leaked minus replay-only control, same fine-tune otherwise): {m:+.3f}")
    cmp = compare(res["clean"], res["leaked, leak undeclared"])
    print("  Eval v3 comparison, clean -> leaked (undeclared):")
    print("    " + report(cmp).replace("\n", "\n    "))
    print(f"  your verdict: {lab.v3_verdict(cmp['_passes'], cmp['_contamination']['base_passes'], cmp['_contamination']['new_passes'], [])}")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "v3_learner.json").write_text(json.dumps(res["clean"]))
    print(f"  wrote {ROOT / 'v3_learner.json'} (your model's Eval v3 result, used by lesson 18.4 and the project)")
    print(f"  (B) {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- part C

def part_lifecycle(lab):
    print("(C) benchmark lifecycle (checked " + L.CHECKED + ")")
    for name in L.REGISTRY:
        print("  " + L.card(name).replace("\n", "\n  "))
        print(f"  -> a v2-passing, contamination-clean comparison on it: {lab.v3_verdict(True, True, True, L.flags(L.REGISTRY[name]))}")


MAIN = [
    "python -m frontierlab.evals.suite_v3.hf score --model runs/m13/main/rlvr-s0/policy --chat --out runs/m18/main/v3/rlvr-s0.json",
    "python -m frontierlab.evals.suite_v3.hf score --model runs/m13/main/sft-s0/policy --chat --out runs/m18/main/v3/sft-s0.json",
    "python -m frontierlab.evals.suite_v3.hf score --model Qwen/Qwen3-1.7B-Base --revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 --out runs/m18/main/v3/base.json",
    "python -m frontierlab.evals.suite_v3.hf compare runs/m18/main/v3/sft-s0.json runs/m18/main/v3/rlvr-s0.json",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["all", "horizon", "contamination", "lifecycle"], default="all")
    ap.add_argument("--runs", type=Path, default=None, help="a local copy of METR's runs.jsonl")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--variant", choices=["cpu", "main", "t4"], default="cpu")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    if a.variant != "cpu":
        for c in MAIN:
            if a.variant == "t4":
                c = c.replace("runs/m13/main", "runs/m13/t4").replace("runs/m18/main", "runs/m18/t4").replace(
                    "Qwen/Qwen3-1.7B-Base --revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
                    "Qwen/Qwen3-0.6B-Base --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
                c += " --n-gsm8k 200" if " score " in c else ""
            print(c)
        return
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    if a.part in ("all", "horizon"):
        part_horizon(lab, a.runs, a.n_boot)
    if a.part in ("all", "contamination"):
        part_contamination(lab)
    if a.part in ("all", "lifecycle"):
        part_lifecycle(lab)


if __name__ == "__main__":
    main()
