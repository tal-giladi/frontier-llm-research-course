"""Module 10 project, debugging task: a colleague's "Data-v1 beats Data-v0" report. It contains two bugs.

    python labs/module-10/project/buggy_report.py            # ~15 min on CPU
    python labs/module-10/project/buggy_report.py --fixed    # the corrected pipeline (after your diagnosis), ~10 min

Their summary: "Data-v1 (Data-v0 plus FineWeb plus a small 'edu-extra' source) lowers held-out loss on Eval v0
clearly, our leakage check found nothing, and the 400-step Data-v1 run trained on exactly the planned tokens per
source." Run it, read what it prints and the run folders it writes, and find both problems before reading the
lesson's reference diagnosis.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import yaml

from frontierlab.data.prepare import DEFAULT_OUT
from frontierlab.datax import arms, leakage
from frontierlab.datax import train as dtrain
from frontierlab.datax.mixture import MixtureSpec, SourceRef
from frontierlab.datax.neardup import decode_docs
from frontierlab.datax.sources import M10_ROOT, load_tokenizer
from frontierlab.data.loader import TokenData

OUT = Path("runs/m10/project-buggy")
COMMON = ["--preset", "toy", "--batch", "16", "--seq", "128", "--lr", "3e-3", "--warmup", "20", "--log-every", "20",
          "--eval-every", "400", "--eval-windows", "32", "--seed", "0"]


def fixed(t0):
    """The corrected pipeline: the leakage index reads the split each source trains on, the 'edu-extra' source
    is dropped (it was Eval v0's own validation split), and the interrupted run is continued in its own folder,
    which resumes the optimizer, the schedule and the mixture stream exactly."""
    tok = load_tokenizer()
    from frontierlab.datax.integrity import eval_v0_window_texts
    from frontierlab.datax import evaluate as ev
    windows = eval_v0_window_texts(DEFAULT_OUT, tok, 128, 128)
    bad = MixtureSpec.load(OUT / "data-v1.json")
    idx = leakage.NgramIndex(13)
    for s in bad.sources:
        idx.add_texts(decode_docs(TokenData(s.split, s.root), tok, 3000))
    idx.finish()
    rep = leakage.leakage_report(idx, windows, 0.5, "eval_v0")
    print(f"leakage check of their recipe, reading each source's own split: {rep['flagged']} of {rep['items']} "
          f"Eval v0 windows flagged")
    v1 = MixtureSpec([SourceRef("edu", str(DEFAULT_OUT), 0.9), SourceRef("web", str(M10_ROOT / "web"), 0.1)],
                     name="data-v1-fixed")
    v1.save(OUT / "data-v1-fixed.json")
    run = OUT / "v1-fixed"
    if not arms.finished(run, 400):
        if not (run / "checkpoint.pt").exists():
            dtrain.main(["--mixture", str(OUT / "data-v1-fixed.json"), "--run", str(run), "--steps", "400",
                         "--stop-after", "200", *COMMON])
        dtrain.main(["--mixture", str(OUT / "data-v1-fixed.json"), "--run", str(run), "--steps", "400", *COMMON])
    sets = {"edu": str(DEFAULT_OUT)}
    a = arms.score(OUT / "v0", sets, windows=128, seq=128, n_lambada=0)
    b = arms.score(run, sets, windows=128, seq=128, n_lambada=0)
    pb = ev.window_paired(a["sets"]["edu"]["losses"], b["sets"]["edu"]["losses"])
    print(f"Eval v0 held-out loss: Data-v0 {np.mean(a['sets']['edu']['losses']):.4f}, fixed Data-v1 "
          f"{np.mean(b['sets']['edu']['losses']):.4f}, paired diff {pb['mean_diff']:+.4f} "
          f"[{pb['ci'][0]:+.4f}, {pb['ci'][1]:+.4f}] (one seed, window-level interval)")
    acc = json.loads((run / "mixture_accounting.json").read_text())
    print("fixed Data-v1 accounting over the whole run:", {k: x["tokens"] for k, x in acc["sources"].items()},
          f"({acc['tokens']} tokens = 400 steps x 16 x 128)")
    print(f"{time.time() - t0:.0f}s")


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    if "--fixed" in sys.argv[1:]:
        return fixed(time.time())
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    v1 = MixtureSpec([SourceRef("edu", str(DEFAULT_OUT), 0.4), SourceRef("web", str(M10_ROOT / "web"), 0.1),
                      SourceRef("edu-extra", str(DEFAULT_OUT), 0.5, split="val")], name="data-v1")
    v0 = MixtureSpec([SourceRef("edu", str(DEFAULT_OUT), 1.0)], name="data-v0")
    v1.save(OUT / "data-v1.json")
    v0.save(OUT / "data-v0.json")

    # "Leakage check": n-gram index over every source of the mixture vs Eval v0's held-out windows
    tok = load_tokenizer()
    idx = leakage.NgramIndex(13)
    for s in v1.sources:
        idx.add_texts(decode_docs(TokenData("train", s.root), tok, 3000))
    idx.finish()
    from frontierlab.datax.integrity import eval_v0_window_texts
    rep = leakage.leakage_report(idx, eval_v0_window_texts(DEFAULT_OUT, tok, 128, 128), 0.5, "eval_v0")
    print(f"leakage check: {rep['flagged']} of {rep['items']} Eval v0 windows flagged -> 'no leakage'")

    # Data-v0 baseline, 200 steps
    if not arms.finished(OUT / "v0", 400):
        dtrain.main(["--mixture", str(OUT / "data-v0.json"), "--run", str(OUT / "v0"), "--steps", "400", *COMMON])
    # Data-v1: the session limit hit at step 200, so the run was continued from its checkpoint in a new folder
    if not (OUT / "v1-part1" / "checkpoint.pt").exists():
        dtrain.main(["--mixture", str(OUT / "data-v1.json"), "--run", str(OUT / "v1-part1"), "--steps", "400",
                     "--stop-after", "200", *COMMON])
    if not (OUT / "v1-part2" / "checkpoint.pt").exists():
        dtrain.main(["--mixture", str(OUT / "data-v1.json"), "--init-from", str(OUT / "v1-part1" / "checkpoint.pt"),
                     "--run", str(OUT / "v1-part2"), "--steps", "200", *COMMON])
    from frontierlab.datax import evaluate as ev
    sets = {"edu": str(DEFAULT_OUT)}
    a = arms.score(OUT / "v0", sets, windows=128, seq=128, n_lambada=0)
    b = arms.score(OUT / "v1-part2", sets, windows=128, seq=128, n_lambada=0)
    pb = ev.window_paired(a["sets"]["edu"]["losses"], b["sets"]["edu"]["losses"])
    print(f"Eval v0 held-out loss: Data-v0 {np.mean(a['sets']['edu']['losses']):.4f}, Data-v1 "
          f"{np.mean(b['sets']['edu']['losses']):.4f}, paired diff {pb['mean_diff']:+.4f} "
          f"[{pb['ci'][0]:+.4f}, {pb['ci'][1]:+.4f}] -> "
          + ("'Data-v1 is clearly better'" if pb["ci"][1] < 0 else "'Data-v1 is not worse'"))
    card = yaml.safe_load((OUT / "v1-part2" / "run_card.yaml").read_text())
    planned = card["datax"]["planned_accounting"]["sources"]
    print("Data-v1 tokens per source (from the run card):", {k: x["tokens"] for k, x in planned.items()},
          "-> 'exactly as planned'")
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
