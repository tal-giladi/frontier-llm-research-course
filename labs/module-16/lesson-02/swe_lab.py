"""Lab 16.2: synthesise software-engineering tasks, split them four ways, and see which held-out scores mean
generalisation.

    python labs/module-16/lesson-02/swe_lab.py                    # free CPU, about 1 minute (+ a 4.1 MB download)
    python labs/module-16/lesson-02/swe_lab.py --shards all       # main path: every SWE-smith shard (CPU)

Part A: SWE-smith-style synthesis on the four toy repositories (five procedural modifications, a candidate is
kept only if it breaks at least one test). Then, for each split unit (instance, function, repository,
family) and 3 split seeds: your ``split_groups``, the sibling leak by function (your ``sibling_leak``), and the
held-out success of two solvers: retrieval (proposes the fix of the most similar training task) and search
(tries single modifications, keeps one that passes 4 visible tests; uses no training data).

Part B: one shard of the real SWE-smith dataset at a pinned revision: for instance-, family- and
repository-level splits, the share of held-out instances whose patch touches a file that a training patch
touches, and whose failing tests include a training instance's failing test (your ``patch_files`` and
``instance_keys``).
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from frontierlab.agents import swetasks as S
from frontierlab.labkit import load_path

HERE = Path(__file__).resolve().parent
SWE, SWE_REV = "SWE-bench/SWE-smith", "ea6d7173829c7ec8fa16c22055699ff2e9188091"
SHARD = "data/train-00002-of-00011.parquet"


def part_a(lab, seeds):
    t0 = time.time()
    tasks, stats = S.synthesise()
    print(f"(A1) synthesis: {stats}  ({time.time() - t0:.1f}s)")
    print(f"     families {dict(Counter(t.family for t in tasks))}")
    print(f"     repositories {dict(Counter(t.repo for t in tasks))}; functions with tasks: "
          f"{len({t.func for t in tasks})}")
    print("(A2) held-out success by split unit (mean over split seeds; per-seed values in brackets)")
    print(f"     {'split by':9s} {'held':>5s} {'sibling leak':>13s} {'retrieval':>22s} {'search':>22s}")
    for by in ("instance", "function", "repo", "family"):
        keys = [t.key[by] for t in tasks]
        leaks, ret, sea, held = [], [], [], []
        for sd in seeds:
            sp = lab.split_groups(keys, 0.25, seed=sd)
            tr = [t for t, s in zip(tasks, sp) if s == "train"]
            he = [t for t, s in zip(tasks, sp) if s == "held"]
            held.append(len(he))
            leaks.append(lab.sibling_leak(sp, [t.func for t in tasks]))
            ret.append(S.solve_and_score(he, tr, "retrieval")["rate"])
            sea.append(S.solve_and_score(he, tr, "search")["rate"])
        fmt = lambda v: f"{np.mean(v):.2f} [" + " ".join(f"{x:.2f}" for x in v) + "]"
        print(f"     {by:9s} {np.mean(held):5.1f} {np.mean(leaks):13.2f} {fmt(ret):>22s} {fmt(sea):>22s}")
    print(f"     part A {time.time() - t0:.0f}s")


def part_b(lab, shards):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    t0 = time.time()
    files = [SHARD] if shards == "one" else [f"data/train-{i:05d}-of-00011.parquet" for i in range(11)]
    ids, patches, f2p = [], [], []
    for f in files:
        p = hf_hub_download(SWE, f, repo_type="dataset", revision=SWE_REV)
        t = pq.read_table(p, columns=["instance_id", "patch", "FAIL_TO_PASS"]).to_pydict()
        ids += t["instance_id"]
        patches += t["patch"]
        f2p += t["FAIL_TO_PASS"]
    keys = [lab.instance_keys(i) for i in ids]
    repos = [k["repository"] for k in keys]
    print(f"(B) SWE-smith @ {SWE_REV[:7]}, {len(files)} shard(s): {len(ids)} instances, {len(set(repos))} repositories, "
          f"families {Counter(k['family'] for k in keys).most_common(4)}")
    for by in ("instance", "family", "repository"):
        sp = lab.split_groups([k[by] for k in keys], 0.25, seed=0)
        tf = {(r, x) for r, p, s in zip(repos, patches, sp) if s == "train" for x in lab.patch_files(p)}
        tt = {(r, x) for r, f, s in zip(repos, f2p, sp) if s == "train" for x in f}
        held = [(r, p, f) for r, p, f, s in zip(repos, patches, f2p, sp) if s == "held"]
        fo = np.mean([bool({(r, x) for x in lab.patch_files(p)} & tf) for r, p, f in held])
        to = np.mean([bool({(r, x) for x in f} & tt) for r, p, f in held])
        both = len({r for r, s in zip(repos, sp) if s == "held"} & {r for r, s in zip(repos, sp) if s == "train"})
        print(f"    split by {by:10s}: held-out {len(held):5d}; patch touches a training file {fo:.3f}; "
              f"shares a failing test with train {to:.3f}; repositories on both sides {both}")
    print(f"    part B {time.time() - t0:.0f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", choices=["one", "all"], default="one")
    ap.add_argument("--part", choices=["a", "b", "both"], default="both")
    a = ap.parse_args(argv)
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    if a.part in ("a", "both"):
        part_a(lab, [0, 1, 2])
    if a.part in ("b", "both"):
        part_b(lab, a.shards)


if __name__ == "__main__":
    main()
