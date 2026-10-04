"""Simulate pipeline schedules: bubble and activation memory per device (lab 09.2, no GPU needed).

    python labs/module-09/lesson-02/simulate.py                       # p = 8 stages, m = 16 micro-batches
    python labs/module-09/lesson-02/simulate.py --p 4 --m 8 --gantt   # small enough to draw
    python labs/module-09/lesson-02/simulate.py --F 1 --B 1.2 --W 0.8 --fb 2.4

For each schedule (GPipe, 1F1B, interleaved 1F1B, the zero-bubble ZB-1P, DualPipe, DualPipeV) it prints the
simulated idle time per device next to the published formula, the bubble ratio, and the peak activations in units
of "one chunk of L/p layers for one micro-batch". Times are in units of one forward chunk. ``--fb`` is the time of
DualPipe's overlapped forward+backward pair: equal to F + B + W when nothing overlaps (the default), smaller when
communication or idle units hide under it. Your ``bubble_ratio`` from lab.py is printed next to the simulator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from frontierlab.dist import schedule as S
from frontierlab.labkit import load_target

lab = load_target(str(Path(__file__).parent / "test_lab.py"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--v", type=int, default=2)
    ap.add_argument("--F", type=float, default=1.0)
    ap.add_argument("--B", type=float, default=1.0, help="input-gradient part of the backward")
    ap.add_argument("--W", type=float, default=1.0, help="weight-gradient part of the backward")
    ap.add_argument("--fb", type=float, default=None)
    ap.add_argument("--gantt", action="store_true")
    a = ap.parse_args()
    t = S.Times(a.F, a.B, a.W, a.fb)
    print(f"p = {a.p} stages, m = {a.m} micro-batches, F = {t.F}, B = {t.B}, W = {t.W} (full backward {t.B_full}), "
          f"F&B = {t.fb if t.fb is not None else t.F + t.B_full}")
    print(f"{'schedule':24s} {'devices':>7s} {'makespan':>9s} {'idle/device (sim)':>18s} {'formula':>8s} "
          f"{'bubble':>7s} {'yours':>7s} {'peak act (max/min)':>18s}")
    for r in S.compare(a.p, a.m, t, a.v):
        mine = ""
        if r["key"] in ("gpipe", "1f1b", "interleaved"):
            try:
                mine = f"{lab.bubble_ratio(r['key'], a.p, a.m, a.v):.3f}"
            except NotImplementedError:
                mine = "TODO"
        idle = max(r["idle"])
        spread = f"{min(r['idle']):.1f}-{idle:.1f}" if max(r["idle"]) - min(r["idle"]) > 1e-9 else f"{idle:.1f}"
        print(f"{r['name']:24s} {r['devices']:7d} {r['makespan']:9.1f} {spread:>18s} {r['formula_idle']:8.1f} "
              f"{r['bubble_ratio']:7.3f} {mine:>7s} {max(r['peak_act']):8.1f} / {min(r['peak_act']):.1f}")
        if a.gantt:
            print(S.gantt(r, 96))
    print("\nDualPipe holds two chunks per device (2x parameters); DualPipeV runs the same p stages on p/2 devices.")
    print("Formulas: (p-1)(F+B+W) for GPipe/1F1B, /v interleaved, (p-1)(F+B+W-2W) ZB-1P, (p/2-1)(F&B+B+W-3W) DualPipe(V).")


if __name__ == "__main__":
    main()
