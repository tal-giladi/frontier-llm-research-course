"""Module 20 project: recompute a capstone package's comparisons with the analysis pieces, and check the package.

    python labs/module-20/project/capstone_project.py --package runs/m20/l201/capstone-qkclip
    CAPSTONE=buggy python labs/module-20/project/capstone_project.py --package runs/m20/l201/capstone-qkclip

Reads each run's cached per-window losses (``runs/<arm>-s<seed>/eval_*.json``, written by the scaffold or by your
own evaluation), pairs them by seed with ``pair``, and prints ``interval`` and ``decide`` for every comparison in
``claim.yaml`` next to what ``results.json`` reports. Then runs the package checker. With ``CAPSTONE=buggy`` the
colleague's pieces are used: compare the two outputs sentence by sentence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from frontierlab.capstone.package import arm_seeds, check_package
from frontierlab.labkit import load_path

HERE = Path(__file__).resolve().parent


def losses(pkg: Path, arm: str, seeds) -> dict[int, list[float]]:
    out = {}
    for s in seeds:
        files = sorted((pkg / "runs" / f"{arm}-s{s}").glob("eval_*.json"))
        if files:
            out[s] = json.loads(files[0].read_text())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", type=Path, required=True)
    a = ap.parse_args(argv)
    P = load_path(str(HERE / ("buggy_capstone.py" if os.environ.get("CAPSTONE") == "buggy" else "pieces.py")))
    claim = yaml.safe_load((a.package / "claim.yaml").read_text())
    reported = {c["name"]: c for c in json.loads((a.package / "results.json").read_text())["comparisons"]}
    for c in claim["comparisons"]:
        la, lb = losses(a.package, c["a"], arm_seeds(claim, c["a"])), losses(a.package, c["b"], arm_seeds(claim, c["b"]))
        if not la or not lb:
            print(f"{c['name']}: no cached per-item losses; skipped")
            continue
        xa, xb, seeds = P.pair(la, lb)
        m, lo, hi = P.interval(xa, xb)
        r = reported.get(c["name"], {})
        print(f"{c['name']:12s} {c['a']} - {c['b']}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] -> "
              f"{P.decide(m, lo, hi, claim['margin'])}   (results.json: {r.get('decision')}, "
              f"[{r.get('ci', [float('nan')] * 2)[0]:+.4f}, {r.get('ci', [float('nan')] * 2)[1]:+.4f}])")
    probs = check_package(a.package)
    print("package:", "SOUND" if not probs else f"{len(probs)} problem(s)")
    for p in probs:
        print("  ", p)
    sys.exit(1 if probs else 0)


if __name__ == "__main__":
    main()
