"""Measure collective bandwidth and exposed communication of data-parallel training (lab 02.3).

    python labs/module-02/lesson-03/run_dp.py                                  # free CPU: 2 gloo ranks
    python labs/module-02/lesson-03/run_dp.py --world 4 --buckets 1 25 100     # more ranks, bucket sweep
    python labs/module-02/lesson-03/run_dp.py --device cuda --world 8 --preset baseline0 --vocab 32768 \\
        --batch 8 --seq 1024 --dtype bf16 --buckets 5 25 100                   # main path, one 8-GPU node

Part 1 times all-reduce at a few buffer sizes and prints bus bandwidth (your ``bus_bandwidth``).
Part 2 times DDP steps at each bucket size and FSDP2 steps, each with and without gradient
synchronisation, and prints the exposed communication with its interval (your ``exposed_comm``) and
the share of the communication that was hidden (your ``overlap_fraction``).

On CPU the collectives are real (gloo) but the bandwidth is that of processes on one machine;
the times say nothing about NVLink or InfiniBand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from frontierlab.labkit import load_target
from frontierlab.perf import dist

lab = load_target(str(Path(__file__).parent / "test_lab.py"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--preset", default="toy")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--buckets", type=float, nargs="+", default=[1.0, 25.0])
    ap.add_argument("--no-fsdp", action="store_true")
    a = ap.parse_args()
    common = dict(device=a.device, preset=a.preset, vocab=a.vocab, batch=a.batch, seq=a.seq,
                  steps=a.steps, dtype=a.dtype)

    sizes = (1, 4, 16, 64)
    ar = dist.spawn(dist.allreduce_worker, a.world, sizes_mb=sizes, device=a.device)[0]
    print(f"1. all-reduce on {a.world} {a.device} ranks")
    for mb in sizes:
        sec = ar[str(mb)]["seconds"]
        print(f"   {mb:4d} MiB  {sec * 1e3:8.2f} ms  bus bandwidth {lab.bus_bandwidth(mb * 2**20, sec, a.world) / 1e9:6.2f} GB/s")
    # bandwidth at the largest size: the best case for one big gradient all-reduce
    best_bw = lab.bus_bandwidth(sizes[-1] * 2**20, ar[str(sizes[-1])]["seconds"], a.world)

    runs = [("ddp", b) for b in a.buckets] + ([] if a.no_fsdp else [("fsdp2", None)])
    print(f"\n2. {a.preset} B={a.batch} T={a.seq} per rank, {a.steps} timed steps per arm")
    print(f"   {'mode':14s} {'buckets':>7s} {'sync ms':>9s} {'nosync ms':>9s} {'exposed ms [95% CI]':>24s} "
          f"{'of step':>7s} {'comm ms':>8s} {'hidden':>6s}")
    for mode, bucket in runs:
        kw = dict(common, mode=mode)
        if bucket is not None:
            kw["bucket_cap_mb"] = bucket
        res = dist.spawn(dist.train_step_worker, a.world, **kw)
        if mode == "ddp":
            sums = [r["param_checksum_after_sync"] for r in res]
            assert max(sums) - min(sums) <= 1e-6 * max(1.0, abs(sums[0])), f"DDP replicas diverged: {sums}"
        # the slowest rank sets the step time
        sync = np.max([r["sync"] for r in res], axis=0)
        nosync = np.max([r["nosync"] for r in res], axis=0)
        e = lab.exposed_comm(sync, nosync)
        grad_bytes = res[0]["grad_bytes"]
        comm = dist.allreduce_bytes_per_rank(grad_bytes, a.world) / best_bw
        name = f"{mode}" + (f" {bucket:g}MiB" if bucket is not None else "")
        nb = res[0].get("n_buckets", "-")
        print(f"   {name:14s} {nb!s:>7s} {np.median(sync) * 1e3:9.1f} {np.median(nosync) * 1e3:9.1f} "
              f"{e['exposed_s'] * 1e3:8.1f} [{e['ci'][0] * 1e3:6.1f}, {e['ci'][1] * 1e3:6.1f}] "
              f"{e['fraction']:7.0%} {comm * 1e3:8.1f} {lab.overlap_fraction(comm, e['exposed_s']):6.0%}")
    print("\n   comm ms = gradient bytes moved per rank / the 64 MiB bus bandwidth above (a lower bound for"
          " one all-reduce of all gradients).\n   For FSDP2 the no-sync arm still all-gathers parameters, so"
          " 'exposed' is the reduce-scatter part only.")


if __name__ == "__main__":
    main()
