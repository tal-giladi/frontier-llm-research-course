"""Lab 10.1, step 3: a stopped and resumed mixture run must be the same run (free CPU, ~2 min).

    python labs/module-10/lesson-01/resume_check.py
    python labs/module-10/lesson-01/resume_check.py --doc-mask

Trains Data-v0 + FineWeb + Wikipedia (60/25/15 by tokens) with ``frontierlab.datax.train`` twice: straight
for ``--steps`` steps, and stopped at half way then restarted with the same command. It prints the
planned per-source token accounting from the run card (known before step 1), the consumed accounting of
both runs, the largest difference between their logged losses and whether the final weights are
bit-identical.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import yaml

from frontierlab.data.prepare import DEFAULT_OUT
from frontierlab.datax import train as dtrain
from frontierlab.datax.mixture import MixtureSpec, SourceRef
from frontierlab.datax.sources import M10_ROOT
from frontierlab.metrics.jsonl import read_jsonl


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("runs/m10/l101-resume"))
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--doc-mask", action="store_true")
    a = ap.parse_args(argv)
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    shutil.rmtree(a.out, ignore_errors=True)
    a.out.mkdir(parents=True)
    spec = MixtureSpec([SourceRef("edu", str(DEFAULT_OUT), 0.60), SourceRef("web", str(M10_ROOT / "web"), 0.25),
                        SourceRef("wiki", str(M10_ROOT / "wiki"), 0.15)], seed=0, name="l101-resume")
    spec.save(a.out / "mix.json")
    common = ["--mixture", str(a.out / "mix.json"), *(["--doc-mask"] if a.doc_mask else []), "--preset", "toy",
              "--steps", str(a.steps), "--batch", "16", "--seq", "128", "--warmup", "10", "--log-every", "5",
              "--eval-every", str(a.steps), "--eval-windows", "16"]
    t0 = time.time()
    m1 = dtrain.main(["--run", str(a.out / "straight"), *common])
    dtrain.main(["--run", str(a.out / "resumed"), *common, "--stop-after", str(a.steps // 2)])
    m2 = dtrain.main(["--run", str(a.out / "resumed"), *common])
    card = yaml.safe_load((a.out / "straight" / "run_card.yaml").read_text())
    planned = card["datax"]["planned_accounting"]
    print("\nplanned accounting (from the run card, before step 1):")
    for k, s in planned["sources"].items():
        print(f"  {k:5s} windows {s['windows']:4d}  tokens {s['tokens']:7d}  share {s['share']:.3f}  epochs {s['epochs']:.5f}"
              f"  documents started {s['documents_started']}")
    print("  requested", planned["requested_weights"], "realised", planned["realised_weights"])
    acc = [json.loads((a.out / r / "mixture_accounting.json").read_text())["sources"] for r in ("straight", "resumed")]
    print("consumed accounting identical (straight vs resumed, and vs planned):", acc[0] == acc[1] == planned["sources"])
    la = {r["step"]: r["loss"] for r in read_jsonl(a.out / "straight" / "metrics.jsonl") if r["split"] == "train"}
    lb = {r["step"]: r["loss"] for r in read_jsonl(a.out / "resumed" / "metrics.jsonl") if r["split"] == "train"}
    print(f"largest logged-loss difference over {len(set(la) & set(lb))} steps: {max(abs(la[s] - lb[s]) for s in la if s in lb)}")
    same = all(torch.equal(x, y) for x, y in zip(m1.state_dict().values(), m2.state_dict().values()))
    print("final weights bit-identical:", same)
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
