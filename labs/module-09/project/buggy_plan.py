"""The Module 9 project's debugging task: a colleague's infrastructure estimate for M9-1T on 2,048 H100s.

    python labs/module-09/project/buggy_plan.py

It prints three headline numbers and a conclusion. Each number is produced by one mistake. Find them (the project
brief has the questions; the diagnosis is in a collapsed block there). Do not fix this file in place: write the
corrected numbers next to the wrong ones in your report and say which check exposes each mistake.
"""

from __future__ import annotations

from frontierlab.calc import param_counts
from frontierlab.dist import goodput as G
from frontierlab.dist import layout as LY

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("plan_1t", Path(__file__).with_name("plan_1t.py"))
plan_1t = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plan_1t)


def main():
    spec = plan_1t.model()
    pc = param_counts(spec)
    world, tokens, mfu, peak = 2048, 15e12, 0.30, 989e12

    # 1. compute
    flops = 6 * pc["total"] * tokens
    gpu_h = flops / (peak * mfu) / 3600
    print(f"1. training compute: 6 x N x D = {flops:.2e} FLOPs -> {gpu_h / 1e6:.1f}M GPU-hours, "
          f"{gpu_h / world / 24:.0f} days on {world} GPUs")

    # 2. reliability
    M = G.mtbf(54 * 24, 419)                          # Llama 3: one unexpected interruption every 3.09 h
    delta, R = 20 / 3600, 10 / 60
    tau = G.young(delta, M)
    print(f"2. job MTBF {M:.1f} h (Llama 3), checkpoint every {tau * 60:.0f} min, goodput "
          f"{G.goodput(tau, delta, M, R):.1%}")

    # 3. expert-parallel communication
    lay = LY.Layout(pp=16, dp=128, ep=64, zero=1, order=("tp", "cp", "ep", "pp", "dp"))
    tr = LY.Train(seq=4096, micro_batch=1, n_micro=32, schedule="dualpipe", optim_bytes=4, act_bytes=1)
    ep_bytes = LY.comm_per_step(spec, lay, tr)["ep"]
    t_ep = ep_bytes / 300e9                           # NVLink bus bandwidth
    print(f"3. EP all-to-all per step: {ep_bytes / 1e9:.0f} GB per GPU at 300 GB/s = {t_ep:.2f} s")

    gp = G.goodput(tau, delta, M, R)
    print(f"\nConclusion: {gpu_h / world / 24 / gp / 365:.1f} years of wall clock on {world:,} GPUs, failures cost "
          f"{1 - gp:.0%} of it, and expert-parallel communication ({t_ep:.2f} s per step) is small enough to ignore."
          " Recommendation: the run is infeasible on this cluster; ask for 16x the GPUs.")


if __name__ == "__main__":
    main()
