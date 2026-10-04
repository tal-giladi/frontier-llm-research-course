"""Summarise the torchtitan runs of lesson 09.3: throughput, MFU, memory per rank, exposed communication.

    python labs/module-09/lesson-03/titan_report.py WORKDIR --skip 10      # after run_titan.sh (main path)
    python labs/module-09/lesson-03/titan_report.py --sample                # the parser on the synthetic sample log

WORKDIR holds ``<config>_rep<k>.log`` (all ranks' output, one line per step per rank) and
``<config>_rep<k>_profiling/traces/iteration_*/rank<r>_trace.json.gz`` written by run_titan.sh.

Per layout: median tokens/s per GPU and MFU over steps after ``--skip`` (warm-up and compilation) with 95%
intervals, pooled over repetitions; peak reserved memory per rank (the largest value each rank printed); and from
each rank's profiler trace, the time NCCL kernels ran with no compute kernel running (exposed communication,
your ``exposed_comm``) as a share of the traced step. Then the paired comparison of the two layouts. Uses your
lab.py by default (LAB_TARGET=solution for the reference).
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from frontierlab.labkit import load_target

lab = load_target(str(Path(__file__).parent / "test_lab.py"))


def read_log(path: Path) -> dict[int, list[dict]]:
    per_rank = defaultdict(list)
    for line in path.read_text(errors="replace").splitlines():
        r = lab.parse_titan_line(line)
        if r is not None:
            per_rank[r["rank"] if r["rank"] is not None else 0].append(r)
    return per_rank


def trace_exposed(path: Path) -> dict:
    with gzip.open(path, "rt") as f:
        ev = json.load(f)["traceEvents"]
    kernels = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"]
    comm = [(e["ts"], e["ts"] + e["dur"]) for e in kernels if "nccl" in e["name"].lower()]
    comp = [(e["ts"], e["ts"] + e["dur"]) for e in kernels if "nccl" not in e["name"].lower()]
    if not kernels:
        return {"exposed_ms": float("nan"), "comm_ms": float("nan"), "span_ms": float("nan")}
    span = max(b for _, b in comm + comp) - min(a for a, _ in comm + comp)
    return {"exposed_ms": lab.exposed_comm(comm, comp) / 1e3, "comm_ms": sum(b - a for a, b in comm) / 1e3,
            "span_ms": span / 1e3}


def summarise(work: Path, skip: int) -> dict:
    out = {}
    logs = sorted(work.glob("*_rep*.log"))
    by_cfg = defaultdict(list)
    for p in logs:
        by_cfg[re.sub(r"_rep\d+$", "", p.stem)].append(p)
    for cfg, paths in by_cfg.items():
        tps, mfu, mem = [], [], defaultdict(float)
        for p in paths:
            for rank, rows in read_log(p).items():
                rows = sorted(rows, key=lambda r: r["step"])
                tps += [r["tps"] for r in rows if r["step"] > skip]
                mfu += [r["mfu"] for r in rows if r["step"] > skip]
                mem[rank] = max(mem[rank], max(r["memory_gib"] for r in rows))
        traces = sorted(work.glob(f"{cfg}_rep*_profiling/traces/iteration_*/rank*_trace.json.gz"))
        exp = [trace_exposed(t) for t in traces]
        out[cfg] = {"tps": lab.summarize(tps, 0) if tps else None, "mfu": lab.summarize(mfu, 0) if mfu else None,
                    "memory_gib_per_rank": dict(sorted(mem.items())), "traces": exp}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", nargs="?", type=Path)
    ap.add_argument("--skip", type=int, default=10)
    ap.add_argument("--sample", action="store_true", help="parse the synthetic sample log shipped with the lab")
    a = ap.parse_args()
    if a.sample:
        rows = read_log(Path(__file__).parent / "sample_titan.log")
        for rank, rs in rows.items():
            s = lab.summarize([r["tps"] for r in rs], 2)
            print(f"rank {rank}: {len(rs)} steps parsed; tps median {s['median']:.0f} [{s['lo']:.0f}, {s['hi']:.0f}] "
                  "(SYNTHETIC sample: checks the parser only)")
        return
    res = summarise(a.work, a.skip)
    for cfg, r in res.items():
        print(f"{cfg}")
        if r["tps"]:
            print(f"  tokens/s per GPU {r['tps']['median']:,.0f} [{r['tps']['lo']:,.0f}, {r['tps']['hi']:,.0f}]  "
                  f"MFU {r['mfu']['median']:.1f}% [{r['mfu']['lo']:.1f}, {r['mfu']['hi']:.1f}]  (n = {r['tps']['n']} rank-steps)")
        print("  peak reserved memory per rank (GiB): " + ", ".join(f"{k}: {v:.1f}" for k, v in r["memory_gib_per_rank"].items()))
        if r["traces"]:
            e = np.array([t["exposed_ms"] for t in r["traces"]])
            s = np.array([t["span_ms"] for t in r["traces"]])
            print(f"  exposed communication in the traced step: {np.nanmedian(e):.1f} ms of {np.nanmedian(s):.1f} ms "
                  f"({np.nanmedian(e / s):.1%}), ranks {np.nanmin(e):.1f}-{np.nanmax(e):.1f} ms")
    names = list(res)
    if len(names) >= 2 and all(res[n]["tps"] for n in names[:2]):
        a_, b_ = names[:2]
        print(f"\n{b_} / {a_} tokens/s: {res[b_]['tps']['median'] / res[a_]['tps']['median']:.3f} "
              "(decide with the contract's rule: intervals must not overlap)")


if __name__ == "__main__":
    main()
