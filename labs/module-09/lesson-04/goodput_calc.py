"""Goodput and checkpoint intervals from Llama 3's interruption log (lab 09.4). Every output is PROJECTED.

    python labs/module-09/lesson-04/goodput_calc.py
    python labs/module-09/lesson-04/goodput_calc.py --delta-s 45 --restart-min 12 --gpus 2048

Inputs (PUBLICLY DOCUMENTED, arXiv 2407.21783 section 3.3.4): 466 interruptions in 54 days, 47 planned and 419
unexpected, about 78% of the unexpected attributed to confirmed or suspected hardware. The job size during the
snapshot is not stated; the script assumes the 16,384-GPU configuration of Table 4 (ASSUMPTION) to derive a per-GPU
failure rate, then scales it to the cluster you plan for.

``--delta-s`` (checkpoint blocking time) and ``--restart-min`` (detect + replace + relaunch + load) default to
ASSUMED values; replace them with your measurements (kill_and_resume.py measures both on your hardware; the project
asks you to scale them to the target cluster and say how).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from frontierlab.dist import goodput as G
from frontierlab.labkit import load_target

lab = load_target(str(Path(__file__).parent / "test_lab.py"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta-s", type=float, default=60.0, help="checkpoint blocking time, seconds (ASSUMED)")
    ap.add_argument("--restart-min", type=float, default=10.0, help="restart cost, minutes (ASSUMED)")
    ap.add_argument("--gpus", type=int, default=16384, help="cluster size to plan for")
    a = ap.parse_args()
    L = G.LLAMA3
    hours = L["days"] * 24
    M_all = G.mtbf(hours, L["interruptions"])
    M_unexp = G.mtbf(hours, L["unexpected"])
    hw = L["unexpected"] * L["hardware_share_of_unexpected"]
    print(f"Llama 3 snapshot: {hours} h, {L['interruptions']} interruptions -> one every {M_all:.2f} h; "
          f"{L['unexpected']} unexpected -> one every {M_unexp:.2f} h ({M_unexp * 60:.0f} min); "
          f"~{hw:.0f} attributed to hardware")
    dev = G.device_mtbf(M_unexp, L["gpus"])
    print(f"Per-GPU MTBF if the job used {L['gpus']:,} GPUs (ASSUMPTION): {dev:,.0f} h = {dev / 8766:.1f} years")
    M = G.job_mtbf(dev, a.gpus)
    d, Rr = a.delta_s / 3600, a.restart_min / 60
    print(f"\nPlanning for {a.gpus:,} GPUs: job MTBF {M:.2f} h; delta = {a.delta_s:.0f} s, R = {a.restart_min:.1f} min")
    print(f"{'interval rule':16s} {'tau (min)':>9s} {'goodput (exact)':>16s} {'goodput (1st order)':>20s} {'yours':>7s}")
    for name, r in G.plan_table(d, M, Rr).items():
        try:
            mine = f"{lab.goodput(r['tau'], d, M, Rr):.4f}"
        except NotImplementedError:
            mine = "TODO"
        print(f"{name:16s} {r['tau'] * 60:9.1f} {r['goodput']:16.4f} {r['first_order_goodput']:20.4f} {mine:>7s}")
    try:
        print(f"your young(): {lab.young(d, M) * 60:.1f} min")
    except NotImplementedError:
        print("your young(): TODO")

    print("\nSensitivity (exact goodput at the optimal interval):")
    print(f"{'':28s}" + "".join(f"{g:>9,d}" for g in (1024, 4096, 16384, 32768, 100000)))
    for label, dd, rr in (("delta 60 s, R 10 min", 60, 10), ("delta 10 s (async), R 10 min", 10, 10),
                          ("delta 60 s, R 0.5 min (elastic)", 60, 0.5), ("delta 10 s, R 0.5 min", 10, 0.5)):
        row = []
        for g in (1024, 4096, 16384, 32768, 100000):
            Mg = G.job_mtbf(dev, g)
            tau = G.optimal_tau(dd / 3600, Mg, rr / 60)
            row.append(G.goodput(tau, dd / 3600, Mg, rr / 60))
        print(f"{label:28s}" + "".join(f"{x:9.3f}" for x in row))
    mc = G.simulate_wall(100.0, G.optimal_tau(d, M, Rr), d, M, Rr, runs=400, seed=0)
    print(f"\nMonte Carlo check (400 simulated 100-hour runs): goodput {100.0 / mc:.4f} vs formula "
          f"{G.goodput(G.optimal_tau(d, M, Rr), d, M, Rr):.4f}")
    print("Planned interruptions (47 in the snapshot) are scheduled: checkpoint right before them and they cost R, not lost work.")


if __name__ == "__main__":
    main()
