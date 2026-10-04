"""A real pipeline across processes: GPipe vs 1F1B, measured bubbles vs the simulator (lab 09.2).

    python labs/module-09/lesson-02/real_pipeline.py                   # free CPU: 4 gloo processes
    python labs/module-09/lesson-02/real_pipeline.py --p 2 --m 8 --steps 8

1. Correctness: the pipelined model's loss and every parameter gradient equal a single-process run on the same
   micro-batches (float64; tied embedding handled by an all-reduce between the first and last stage).
2. Measurement: ``--steps`` optimizer steps of each schedule (alternating GPipe, 1F1B, GPipe, 1F1B, ...). For
   each rank: busy time (forward and backward compute) and the bubble ``1 - busy / step time`` (your
   ``measured_bubble``), where the step time is the slowest rank's wall time (the next step cannot start
   earlier; the simulator uses the same definition); peak live micro-batches (activation memory) per rank.
3. Prediction: the simulator fed with the *measured* mean forward and backward times per micro-batch.

These are CPU processes on one machine (``gloo``). Bubbles come from the schedule's dependencies, which are the
same on GPUs; the absolute times, the communication cost and the cross-rank interference are CPU-specific (all
ranks share the machine's cores and memory bandwidth).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from frontierlab.dist import schedule as S
from frontierlab.dist.pipeline import pipeline_worker
from frontierlab.labkit import load_target
from frontierlab.perf.dist import spawn
from frontierlab.stats import bootstrap_ci

lab = load_target(str(Path(__file__).parent / "test_lab.py"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=4)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6, help="timed steps per round (the first one is warm-up)")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--micro-batch", type=int, default=4)
    a = ap.parse_args()

    print(f"1. correctness: {a.p} stages, 4 micro-batches, float64")
    res = spawn(pipeline_worker, a.p, schedule="1f1b", m=4, steps=1, dtype="float64", check=True, hidden=32,
                vocab=61, seq=16, layers=2 * a.p)
    gerr = max(r["grad_max_abs_err"] for r in res)
    print(f"   max |grad diff| over all stages {gerr:.1e}, |loss diff| {res[-1]['loss_abs_err']:.1e}")
    assert gerr < 1e-12 and res[-1]["loss_abs_err"] < 1e-12, "the pipeline does not compute the same gradients"

    print(f"\n2. {a.p} stages of 2 layers, m = {a.m} micro-batches of {a.micro_batch} x {a.seq} tokens, "
          f"hidden {a.hidden}, float32, {a.rounds} rounds x {a.steps - 1} timed steps per schedule")
    data = {"gpipe": [], "1f1b": []}
    for _ in range(a.rounds):
        for name in data:
            data[name].append(spawn(pipeline_worker, a.p, schedule=name, m=a.m, steps=a.steps, hidden=a.hidden,
                                    seq=a.seq, micro_batch=a.micro_batch))
    sim_rows = {}
    for name, runs in data.items():
        busy = [np.concatenate([r[k]["busy"][1:] for r in runs]) for k in range(a.p)]
        wall = [np.concatenate([r[k]["wall"][1:] for r in runs]) for k in range(a.p)]
        f = np.mean([np.mean(r[k]["f"][1:]) for r in runs for k in range(a.p)])
        b = np.mean([np.mean(r[k]["b"][1:]) for r in runs for k in range(a.p)])
        step_wall = np.max(np.stack(wall), axis=0)          # the step ends when the slowest rank ends
        bubbles = []
        for k in range(a.p):
            mb = lab.measured_bubble(busy[k], step_wall)
            _, lo, hi = bootstrap_ci(1 - busy[k] / step_wall, stat=np.median, n_boot=2000)
            bubbles.append((mb, lo, hi))
        sch = S.gpipe(a.p, a.m) if name == "gpipe" else S.one_f_one_b(a.p, a.m)
        sim = S.simulate(sch, S.Times(F=f, B=b, W=0.0))
        sim_rows[name] = sim
        peak = [runs[0][k]["peak_live_microbatches"] for k in range(a.p)]
        slow = np.median(step_wall)
        print(f"   {name:5s} measured bubble per rank: " + ", ".join(f"{m:.2f} [{lo:.2f}, {hi:.2f}]" for m, lo, hi in bubbles))
        print(f"         slowest-rank step {slow * 1e3:.0f} ms; mean F {f * 1e3:.1f} ms, B {b * 1e3:.1f} ms per micro-batch")
        print(f"         simulator with those F and B: bubble {sim['bubble_ratio']:.2f}, step {sim['makespan'] * 1e3:.0f} ms;"
              f" formula (p-1)/(m+p-1) = {lab.bubble_ratio(name, a.p, a.m):.2f}")
        print(f"         peak live micro-batches per rank (activation memory): {peak}")
    print("\n   Decision rule (contract): the schedules differ in bubble only if their intervals do not overlap; they "
          "differ in memory by the peak counts above.")


if __name__ == "__main__":
    main()
