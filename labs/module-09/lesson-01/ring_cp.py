"""Ring attention across processes: correctness first, then contiguous vs load-balanced sharding (lab 09.1).

    python labs/module-09/lesson-01/ring_cp.py                      # free CPU: gloo, 2 and 4 ranks
    python labs/module-09/lesson-01/ring_cp.py --world 4 --T 8192 --rounds 3
    python labs/module-09/lesson-01/ring_cp.py --device cuda --world 8 --T 32768 --heads 32 --dim 128   # main path

Part 1 (correctness): ring attention on 2 and 4 ranks in float64 against single-device attention on the whole
sequence — output and the gradients of q, k and v, for both shardings. Every difference must be below 1e-12.

Part 2 (the comparison in the experiment contract): forward + backward of ring attention in float32 on ``--world``
ranks for a ``--T``-token causal sequence, contiguous vs load-balanced sharding, in alternating rounds. Reported per
arm: the slowest rank's step time (median and 95% interval), each rank's share of the causal work, and how long
ranks waited for the next K/V block. CPU gloo collectives are real collectives; their times say nothing about
NVLink or InfiniBand.
"""

from __future__ import annotations

import argparse

import numpy as np

from frontierlab.dist import ring_attention as RA
from frontierlab.perf.dist import spawn
from frontierlab.perf.timing import Timing, speedup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--device", default="cpu", help="cuda: NCCL, one GPU per rank, bf16 (main path)")
    a = ap.parse_args()

    print("1. correctness (float64, ring vs single-device attention, GQA 4 query heads / 2 KV heads)")
    for world in (2, 4):
        for balanced in (False, True):
            res = spawn(RA.equivalence_worker, world, T=8 * world, balanced=balanced)
            worst = max(max(r["out"], r["dq"], r["dk"], r["dv"]) for r in res)
            print(f"   world {world}  {'balanced  ' if balanced else 'contiguous'}  max |diff| over out, dq, dk, dv: {worst:.1e}")
            assert worst < 1e-12, "ring attention disagrees with single-device attention"

    print(f"\n2. {a.world} ranks ({a.device}), T = {a.T}, {a.heads} heads x {a.dim}, "
          f"{'float32' if a.device == 'cpu' else 'bf16'}, forward + backward")
    arms = {"contiguous": Timing(), "balanced": Timing()}
    detail = {}
    for _ in range(a.rounds):
        for name in arms:
            res = spawn(RA.timing_worker, a.world, T=a.T, H=a.heads, KV=a.heads, d=a.dim, balanced=name == "balanced",
                        repeats=a.repeats, device=a.device)
            slowest = np.max([r["samples"] for r in res], axis=0)
            arms[name].samples.extend(slowest.tolist())
            detail[name] = res
    total = a.T * (a.T + 1) // 2
    for name, t in arms.items():
        res = detail[name]
        shares = [r["pairs"] / total for r in res]
        print(f"   {name:10s}  slowest-rank step {t}")
        print(f"              causal work per rank {['%.0f%%' % (100 * s) for s in shares]}, "
              f"forward compute per rank {[round(r['fwd_compute_s'] * 1e3) for r in res]} ms, "
              f"forward wait per rank {[round(r['fwd_wait_s'] * 1e3) for r in res]} ms")
    s = speedup(arms["contiguous"], arms["balanced"])
    print(f"\n   balanced vs contiguous: {s['speedup']:.2f}x faster [95% CI {s['ci'][0]:.2f}, {s['ci'][1]:.2f}]"
          f" (ideal from the work split: {max(RA.work_per_rank(a.T, a.world, False)) / max(RA.work_per_rank(a.T, a.world, True)):.2f}x)")
    print("   Decision rule (contract): adopt balanced sharding if the interval lies entirely above 1.")


if __name__ == "__main__":
    main()
