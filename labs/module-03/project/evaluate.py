"""Score finished Module 3 runs with Eval Suite v0 (held-out windows + LAMBADA), keeping every item.

    python labs/module-03/project/evaluate.py runs/m03/cpu/*-s0 runs/m03/cpu/*-s1
    python labs/module-03/project/evaluate.py runs/m03/main/* --device cuda --bf16
    python labs/module-03/project/evaluate.py RUN --split test        # ONCE, after the memo's decision is written

Writes RUN/eval_v0_<split>.json in the Module 1 format (per-window losses, per-passage LAMBADA scores,
suite pins) so the Module 1 noise floor and these runs are on the same footing. Works for every Module 3
attention kind (importing m03 registers them). Skips runs that already have the file.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m03  # noqa: E402
from frontierlab.data.loader import TokenData  # noqa: E402
from frontierlab.evals import suite_v0  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--lambada", type=int, default=None, help="first N passages only (same N for every run)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    data = TokenData(a.split)
    suite_v0.download_lambada()
    enc = suite_v0.data_v0_encoder()
    for run in a.runs:
        out = run / f"eval_v0_{a.split}.json"
        if out.exists() and not a.force:
            print(f"{run}: {out.name} exists, skipping")
            continue
        if not (run / "checkpoint.pt").exists():
            print(f"{run}: no checkpoint, skipping")
            continue
        model, card = m03.load_run(run, a.device)
        T = int(card["args"]["seq"])
        t0 = time.perf_counter()
        res = suite_v0.run_suite(model, data, enc, n_windows=a.windows, T=T, n_lambada=a.lambada, device=a.device,
                                 pad_id=data.meta.get("eot_id", 0),
                                 autocast_dtype=torch.bfloat16 if a.bf16 else None)
        res["run"], res["seconds"] = str(run), round(time.perf_counter() - t0, 1)
        out.write_text(json.dumps(res))
        s = suite_v0.summarize(res)
        print(f"{run}  held-out {s['heldout_loss'][0]:.4f}  LAMBADA logprob {s['lambada_target_logprob'][0]:.3f}  "
              f"acc {s['lambada_acc'][0]:.4f}  ({res['seconds']} s)")


if __name__ == "__main__":
    main()
