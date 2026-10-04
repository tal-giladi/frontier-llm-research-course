"""Plan parallel layouts with the formulas of lesson 09.1 (no GPU needed; every number is an ESTIMATE or PROJECTED).

    python labs/module-09/lesson-01/plan_layouts.py                  # Llama 3 405B, DeepSeek-V3, Baseline-0
    python labs/module-09/lesson-01/plan_layouts.py --mfu 0.38       # change the assumed MFU

Part 1 replays the three Llama 3 405B rows of Table 4 (arXiv 2407.21783) and shows what the planner says about
memory and communication per GPU, then the same layout with the rank order reversed. Part 2 plans DeepSeek-V3's
documented layout (section 3.2: PP16, EP64 over 8 nodes, ZeRO-1, no TP) on H800 links. Part 3 is your own
model: Baseline-0 on one 8-GPU node, four candidate layouts. Your ``group_nodes`` from lab.py is checked against
the planner's placement on every layout printed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from frontierlab.calc import from_course, from_hf
from frontierlab.dist import layout as LY
from frontierlab.labkit import load_target
from frontierlab.model import baseline0

lab = load_target(str(Path(__file__).parent / "test_lab.py"))


def show(spec, lay, tr, cl, mfu):
    r = LY.plan(spec, lay, tr, cl, mfu)
    print(LY.format_plan(r))
    sizes = {"tp": lay.tp, "cp": lay.cp, "pp": lay.pp, "dp": lay.dp}
    order = [d for d in lay.order if d != "ep"]
    if lay.ep == 1:
        for d in ("tp", "cp", "pp", "dp"):
            mine = lab.group_nodes(order, sizes, [d], cl.gpus_per_node)
            assert mine == r["nodes_spanned"][d], f"your group_nodes says {mine} nodes for {d}, the planner {r['nodes_spanned'][d]}"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mfu", type=float, default=0.40, help="assumed MFU for the compute time (PROJECTED)")
    a = ap.parse_args()

    print("1. Llama 3 405B, Table 4 layouts, order [TP, CP, PP, DP] (innermost first), 8 H100 per node")
    ll = LY.llama3_405b()
    rows = [(LY.Layout(tp=8, cp=1, pp=16, dp=64), LY.Train(seq=8192, micro_batch=1, n_micro=32, schedule="interleaved")),
            (LY.Layout(tp=8, cp=1, pp=16, dp=128), LY.Train(seq=8192, micro_batch=1, n_micro=16, schedule="interleaved")),
            (LY.Layout(tp=8, cp=16, pp=16, dp=8), LY.Train(seq=131072, micro_batch=1, n_micro=16, schedule="interleaved"))]
    for lay, tr in rows:
        lay = LY.with_(lay, vpp=2)
        show(ll, lay, tr, LY.H100_NODE, a.mfu)
        full = LY.plan(ll, lay, LY.Train(**{**tr.__dict__, "recompute": "full"}), LY.H100_NODE, a.mfu)
        print(f"  with full activation recomputation: {full['memory']['total'] / 1e9:.1f} GB per GPU")
        print()
    print("   same as row 2, rank order reversed ([DP, PP, CP, TP]): TP groups now leave the node")
    show(ll, LY.Layout(tp=8, pp=16, dp=128, vpp=2, order=("dp", "ep", "pp", "cp", "tp")), rows[1][1], LY.H100_NODE, a.mfu)

    print("\n2. DeepSeek-V3 (section 3.2): PP16 x EP64 (over 8 nodes) x ZeRO-1 DP, no TP, H800 links, DualPipe")
    ds = from_hf("deepseek-ai/DeepSeek-V3")
    tr = LY.Train(seq=4096, micro_batch=1, n_micro=120, schedule="dualpipe", optim_bytes=4)   # 15,360 x 4K per step
    show(ds, LY.Layout(pp=16, dp=128, ep=64, zero=1, order=("tp", "cp", "ep", "pp", "dp")), tr, LY.H800_NODE, a.mfu)
    print("   the same layout with activations cached in FP8 (section 3.3.3 caches linear inputs in FP8):")
    show(ds, LY.Layout(pp=16, dp=128, ep=64, zero=1, order=("tp", "cp", "ep", "pp", "dp")),
         LY.Train(**{**tr.__dict__, "act_bytes": 1}), LY.H800_NODE, a.mfu)

    print("\n3. Baseline-0 on one 8x H100 node, global batch 64 x 1,024 tokens")
    b0 = from_course(baseline0())
    for lay in (LY.Layout(dp=8, zero=0), LY.Layout(dp=8, zero=3), LY.Layout(tp=2, dp=4, zero=3),
                LY.Layout(pp=2, dp=4, zero=1)):
        n_micro = 64 // lay.dp
        sched = "1f1b" if lay.pp > 1 else "1f1b"
        show(b0, lay, LY.Train(seq=1024, micro_batch=1, n_micro=n_micro, schedule=sched), LY.H100_NODE, a.mfu)
    print("\nAll memory figures are ESTIMATES from the lesson's formulas; times are PROJECTED from assumed bandwidths"
          " (H100: 300 GB/s in a node, 40 GB/s per GPU between nodes) and the assumed MFU. Measure before deciding.")


if __name__ == "__main__":
    main()
