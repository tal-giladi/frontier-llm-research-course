"""Held-out loss of several Module 5 runs on the same fixed windows, paired against the first run.

    python labs/module-05/heldout_compare.py runs/m05/l51/cpu/b0-s0 runs/m05/l51/cpu/hybrid-kda-s0 ... \\
        --seq 256 --windows 256 --margin 0.03

Every run is scored on the same validation windows (``frontierlab.evals.heldout.window_losses``, Eval v0's
first component), so differences are paired by window and get a paired-bootstrap 95% interval (lesson
01.4). The decision column applies the rule the lab's contract states: "within margin" when the whole
interval lies inside +/- margin, "worse" / "better" when it lies entirely outside zero on that side,
"inconclusive" otherwise. Also prints parameters, training FLOPs per token and tokens from the run cards
so you can check budget parity before reading any loss.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m05  # noqa: E402
from frontierlab.data.loader import TokenData  # noqa: E402
from frontierlab.evals.heldout import window_losses  # noqa: E402
from frontierlab.stats import paired_bootstrap  # noqa: E402


def verdict(ci, margin):
    lo, hi = ci
    if -margin < lo and hi < margin:
        return "within margin"
    if lo > 0:
        return "worse (higher loss)"
    if hi < 0:
        return "better (lower loss)"
    return "inconclusive"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--split", default="val")
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    data = TokenData(a.split)
    rows = []
    for r in a.runs:
        model, card = m05.load_run(r, a.device)
        losses = window_losses(model, data, a.windows, a.seq, device=a.device)
        b = (card or {}).get("budget", {})
        rows.append({"run": r, "attention": model.config.attention, "losses": losses,
                     "params": (card or {}).get("params", {}).get("non_embedding"),
                     "flops_per_token": b.get("train_flops", 0) / max(1, b.get("tokens", 1)), "tokens": b.get("tokens")})
    base = rows[0]
    print(f"{a.windows} {a.split} windows of {a.seq} tokens; differences are run - {Path(base['run']).name}, "
          f"paired by window, 95% bootstrap CI; margin {a.margin}")
    print(f"{'run':<28s} {'attention':<8s} {'non-emb':>9s} {'MFLOP/tok':>9s} {'tokens':>10s} {'loss':>7s} "
          f"{'diff':>8s} {'95% CI':>20s}  decision")
    for r in rows:
        mean = sum(r["losses"]) / len(r["losses"])
        line = (f"{Path(r['run']).name:<28s} {r['attention']:<8s} {(r['params'] or 0) / 1e6:8.3f}M "
                f"{r['flops_per_token'] / 1e6:9.2f} {r['tokens'] or 0:>10,d} {mean:7.4f}")
        if r is not base:
            pb = paired_bootstrap(r["losses"], base["losses"])
            r["paired"] = pb
            line += f" {pb['mean_diff']:+8.4f} [{pb['ci'][0]:+.4f}, {pb['ci'][1]:+.4f}]  {verdict(pb['ci'], a.margin)}"
        print(line)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
