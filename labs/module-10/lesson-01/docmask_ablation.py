"""Lab 10.1, step 4: packing with and without document masking, matched by seed (free CPU defaults).

    python labs/module-10/lesson-01/docmask_ablation.py                       # 2 arms x 3 seeds, ~25 min on CPU
    python labs/module-10/lesson-01/docmask_ablation.py --analyze-only

Both arms run ``frontierlab.datax.train`` on Data-v0 with the loop's own random windows; the masked arm adds
``--doc-mask``. With the same ``--seed`` the two arms see the same windows in the same order and start from
the same weights (gqa and gqa-docmask have identical parameters), so the per-seed difference isolates the mask.

Scoring (``frontierlab.datax.packing.doc_losses``): the first ``--seq`` tokens of each of the first 200
validation documents, every document scored *on its own*, where masked and unmasked attention see exactly
the same context. Also reported: Eval v0's fixed windows, each arm scored the way it was trained, and the
measured training throughput of each arm (the mask path of SDPA is slower on CPU than the is_causal path).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from frontierlab.data.loader import TokenData
from frontierlab.datax import evaluate as ev
from frontierlab.datax import packing
from frontierlab.datax import train as dtrain
from frontierlab.evals.heldout import window_losses
from frontierlab.labkit import load_path
from frontierlab.metrics.jsonl import read_jsonl

HERE = Path(__file__).resolve().parent
VARIANTS = {
    "cpu": dict(preset="toy", steps=300, batch=4, seq=512, lr=3e-3, warmup=30, seeds=[0, 1, 2], device="cpu"),
    "t4": dict(preset="pilot-10m", steps=2000, batch=16, seq=1024, lr=3e-3, warmup=100, seeds=[0, 1, 2], device="cuda"),
    "main": dict(preset="baseline0", steps=4000, batch=32, seq=2048, lr=3e-3, warmup=200, seeds=[0, 1, 2], device="cuda"),
}


def train_arm(out: Path, v: dict, seed: int, masked: bool) -> Path:
    run = out / f"{'mask' if masked else 'nomask'}-s{seed}"
    if (run / "checkpoint.pt").exists() and any(r.get("step") == v["steps"] and r.get("split") == "val"
                                                for r in read_jsonl(run / "metrics.jsonl")):
        return run
    args = ["--run", str(run), "--preset", v["preset"], "--steps", str(v["steps"]), "--batch", str(v["batch"]),
            "--seq", str(v["seq"]), "--lr", str(v["lr"]), "--warmup", str(v["warmup"]), "--seed", str(seed),
            "--eval-every", str(v["steps"]), "--eval-windows", "64", "--log-every", "50", "--device", v["device"],
            "--question", "does document masking change per-document loss at equal tokens? (lesson 10.1)"]
    if v["device"] == "cuda":
        args += ["--dtype", "bf16"]
    if masked:
        args = ["--doc-mask"] + args
    t0 = time.time()
    dtrain.main(args)
    (run / "wall_seconds.txt").write_text(f"{time.time() - t0:.1f}\n")
    return run


def score(run: Path, seq: int, device: str) -> dict:
    path = run / "eval_docmask.json"
    if path.exists():
        return json.loads(path.read_text())
    model, cfg, _ = ev.load_run(run, device)
    val = TokenData("val")
    doc = packing.doc_losses(model, val, max_docs=200, T=seq, device=device)
    if cfg.attention == "gqa-docmask":
        win = packing.window_losses_docmask(model, val, 64, seq, device=device)
    else:
        win = window_losses(model, val, 64, seq, device=device)
    tps = [r["tok_per_s"] for r in read_jsonl(run / "metrics.jsonl") if r.get("split") == "train"][1:]
    res = {"doc": doc, "windows": win, "tok_per_s": float(np.median(tps)) if tps else None}
    path.write_text(json.dumps(res))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--out", type=Path, default=Path("runs/m10/l101"))
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--margin", type=float, default=0.01)
    a = ap.parse_args(argv)
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    v = VARIANTS[a.variant]
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    t0 = time.time()
    res = {}
    for seed in v["seeds"]:
        for masked in (False, True):
            run = a.out / f"{'mask' if masked else 'nomask'}-s{seed}"
            if not a.analyze_only:
                train_arm(a.out, v, seed, masked)
            res[(masked, seed)] = score(run, v["seq"], v["device"])
    # per-document means per seed, paired by seed
    seeds = v["seeds"]
    a_doc = [float(np.mean(res[(False, s)]["doc"])) for s in seeds]
    b_doc = [float(np.mean(res[(True, s)]["doc"])) for s in seeds]
    a_win = [float(np.mean(res[(False, s)]["windows"])) for s in seeds]
    b_win = [float(np.mean(res[(True, s)]["windows"])) for s in seeds]
    sl = ev.seed_level(a_doc, b_doc)
    print(f"per-document loss (each validation document alone, first {v['seq']} tokens, 200 documents)")
    for s, x, y in zip(seeds, a_doc, b_doc):
        pb = ev.window_paired(res[(False, s)]["doc"], res[(True, s)]["doc"])
        print(f"  seed {s}: no mask {x:.4f}  mask {y:.4f}  diff {y - x:+.4f}  paired over documents "
              f"[{pb['ci'][0]:+.4f}, {pb['ci'][1]:+.4f}]")
    print(f"  seed-level (mask - no mask): {sl['mean_diff']:+.4f}  95% CI [{sl['ci'][0]:+.4f}, {sl['ci'][1]:+.4f}]"
          f"  (n = {sl['n']} seeds)")
    print(f"  seed std of the no-mask arm: {np.std(a_doc, ddof=1):.4f}")
    sw = ev.seed_level(a_win, b_win)
    print(f"Eval v0 windows, each arm with its own attention: {sw['mean_diff']:+.4f}  95% CI [{sw['ci'][0]:+.4f}, {sw['ci'][1]:+.4f}]")
    tp_a = [res[(False, s)]["tok_per_s"] for s in seeds]
    tp_b = [res[(True, s)]["tok_per_s"] for s in seeds]
    print(f"training throughput (median tok/s per run): no mask {tp_a}, mask {tp_b}")
    print(f"decision (non-inferiority margin {a.margin}): {lab.decide_noninferior(sl['ci'], a.margin)}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
