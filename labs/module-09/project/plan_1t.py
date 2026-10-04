"""Module 9 project: an infrastructure plan for a 1T-total / ~41B-active MoE model on a stated cluster.

    python labs/module-09/project/plan_1t.py                         # defaults: published numbers + assumptions
    python labs/module-09/project/plan_1t.py --inputs my_inputs.json # your measured values from 09.3 and 09.4
    python labs/module-09/project/plan_1t.py --write-template my_inputs.json

The model ("M9-1T") is Kimi K2's published configuration (moonshotai/Kimi-K2-Instruct config.json snapshot: 61
layers, width 7,168, MLA, 384 routed experts of width 2,048) with 10 routed + 2 shared experts per token instead of
8 + 1, which gives 1.03T total and 40.8B active parameters. It is a course construct, not a released model.

Every input carries a label: MEASURED (you ran it; say on what), PUBLISHED (cite it), ASSUMED (say why). The script
prints each number of the plan with the labels of the inputs it depends on, so a reviewer can see what is
measured, what is borrowed and what is guessed. Change the inputs, not the code.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from frontierlab.calc import flops_per_token, from_hf, param_counts
from frontierlab.dist import goodput as G
from frontierlab.dist import layout as LY

DEFAULTS = {
    "cluster_gpus": {"value": 2048, "label": "ASSUMED", "note": "the stated cluster: 256 nodes x 8 H100 SXM 80 GB"},
    "gpus_per_node": {"value": 8, "label": "PUBLISHED", "note": "H100 HGX node"},
    "peak_flops": {"value": 989e12, "label": "PUBLISHED", "note": "H100 SXM dense BF16 (NVIDIA datasheet; company claim)"},
    "intra_bw": {"value": 300e9, "label": "ASSUMED",
                 "note": "NVLink all-reduce bus bandwidth; replace with run_titan.sh step 3 (09.3) on your node"},
    "inter_bw": {"value": 40e9, "label": "ASSUMED", "note": "per-GPU IB 400 Gb/s at 80% efficiency; measure with nccl-tests across 2 nodes"},
    "mfu": {"value": 0.30, "label": "ASSUMED",
            "note": "MoE with EP all-to-all; replace with the MFU you measured in 09.3 and discount it for EP (say how)"},
    "tokens": {"value": 15e12, "label": "ASSUMED", "note": "training tokens of the run (Kimi K2 reports 15.5T; arXiv 2507.20534)"},
    "seq": {"value": 4096, "label": "ASSUMED", "note": "pre-training sequence length before long-context extension"},
    "global_batch_tokens": {"value": 16.8e6, "label": "ASSUMED", "note": "about 16M tokens per step, as Llama 3 (Table 4)"},
    "ckpt_write_GBps_per_node": {"value": 2.0, "label": "ASSUMED",
                                 "note": "per-node write bandwidth to shared storage; 09.4 measures local-disk write on your machine"},
    "storage_aggregate_GBps": {"value": 2000.0, "label": "PUBLISHED",
                               "note": "Llama 3 Tectonic sustained 2 TB/s (section 3.3.1) as a stand-in for the storage tier"},
    "async_blocking_fraction": {"value": 0.1, "label": "ASSUMED",
                                "note": "async save blocks for the copy to host; 09.4 measures the async/sync ratio on CPU"},
    "restart_min": {"value": 10.0, "label": "ASSUMED",
                    "note": "detect + exclude node + relaunch + load; 09.4 measures relaunch + load at small scale"},
    "gpu_mtbf_h": {"value": None, "label": "PUBLISHED-DERIVED",
                   "note": "default: Llama 3's 419 unexpected interruptions in 54 days on an assumed 16,384 GPUs"},
    "usd_per_gpu_h": {"value": 2.5, "label": "ASSUMED", "note": "rental price; check current offers"},
}


def model():
    k = from_hf("moonshotai/Kimi-K2-Instruct")
    return dataclasses.replace(k, name="M9-1T (course construct)", top_k=10, n_shared=2,
                               shared_intermediate=2 * k.expert_intermediate)


def load_inputs(path: Path | None) -> dict:
    inp = json.loads(json.dumps(DEFAULTS))
    if path is not None:
        for k, v in json.loads(path.read_text()).items():
            inp[k] = v if isinstance(v, dict) else {"value": v, "label": "MEASURED", "note": "from your file"}
    if inp["gpu_mtbf_h"]["value"] is None:
        L = G.LLAMA3
        inp["gpu_mtbf_h"]["value"] = G.device_mtbf(G.mtbf(L["days"] * 24, L["unexpected"]), L["gpus"])
    return inp


def candidates(world: int, gpn: int):
    return [
        ("A: PP16 x EP64 (8 nodes) x ZeRO-1, no TP (DeepSeek-V3 style)",
         LY.Layout(pp=16, dp=world // 16, ep=64, zero=1, order=("tp", "cp", "ep", "pp", "dp")), "dualpipe"),
        ("B: PP16 x EP32 (4 nodes) x ZeRO-1, no TP",
         LY.Layout(pp=16, dp=world // 16, ep=32, zero=1, order=("tp", "cp", "ep", "pp", "dp")), "dualpipe"),
        ("C: TP8 (in node) x PP8 x EP32 x ZeRO-1",
         LY.Layout(tp=8, pp=8, dp=world // 64, ep=32, zero=1), "1f1b"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, default=None)
    ap.add_argument("--write-template", type=Path, default=None)
    ap.add_argument("--act-bytes", type=int, default=1, help="1 = FP8 activation caching (lesson 08.2), 2 = BF16")
    a = ap.parse_args()
    if a.write_template:
        a.write_template.write_text(json.dumps(DEFAULTS, indent=2))
        print(f"template written to {a.write_template}; edit values and labels, then pass it with --inputs")
        return
    inp = load_inputs(a.inputs)
    v = {k: x["value"] for k, x in inp.items()}
    spec = model()
    pc = param_counts(spec)
    fpt = flops_per_token(spec, int(v["seq"]))
    print(f"Model {spec.name}: {pc['total'] / 1e12:.3f}T total, {pc['active_with_embedding'] / 1e9:.1f}B active; "
          f"{fpt / 1e9:.0f} GFLOPs per training token at T = {v['seq']} (frontierlab.calc)")
    print("\nInputs:")
    for k, x in inp.items():
        print(f"  {k:26s} {x['value']!s:>14s}  [{x['label']}] {x['note']}")
    cl = LY.Cluster(name="stated", gpus_per_node=int(v["gpus_per_node"]), peak_flops=v["peak_flops"],
                    intra_bw=v["intra_bw"], inter_bw=v["inter_bw"])
    world = int(v["cluster_gpus"])
    print(f"\nLayouts on {world} GPUs (memory = ESTIMATE; times PROJECTED from intra_bw/inter_bw/mfu above):")
    for name, lay, sched in candidates(world, cl.gpus_per_node):
        per_dp = v["global_batch_tokens"] / lay.dp
        n_micro = max(1, round(per_dp / v["seq"]))
        tr = LY.Train(seq=int(v["seq"]), micro_batch=1, n_micro=n_micro, schedule=sched, optim_bytes=4,
                      act_bytes=a.act_bytes)
        r = LY.plan(spec, lay, tr, cl, v["mfu"])
        print(f"\n{name}")
        print(LY.format_plan(r))
    total_flops = fpt * v["tokens"]
    gpu_h = total_flops / (v["peak_flops"] * v["mfu"]) / 3600
    M = G.job_mtbf(v["gpu_mtbf_h"], world)
    ckpt_bytes = pc["total"] * (4 + 4 + 2)            # fp32 master, bf16 moments x2, bf16 weights
    write_bw = min(v["ckpt_write_GBps_per_node"] * world / v["gpus_per_node"], v["storage_aggregate_GBps"]) * 1e9
    delta_sync = ckpt_bytes / write_bw
    delta = delta_sync * v["async_blocking_fraction"]
    Rh = v["restart_min"] / 60
    tau = G.optimal_tau(delta / 3600, M, Rh)
    gp = G.goodput(tau, delta / 3600, M, Rh)
    wall_days = gpu_h / world / gp / 24
    print("\nRun totals:")
    print(f"  training compute {total_flops:.2e} FLOPs -> {gpu_h / 1e6:.2f}M GPU-hours at MFU {v['mfu']} "
          f"[{inp['mfu']['label']}, {inp['tokens']['label']} tokens]")
    print(f"  checkpoint {ckpt_bytes / 1e12:.1f} TB; write bandwidth {write_bw / 1e9:.0f} GB/s "
          f"[{inp['ckpt_write_GBps_per_node']['label']}/{inp['storage_aggregate_GBps']['label']}] -> sync {delta_sync:.0f} s, "
          f"blocking {delta:.0f} s with async [{inp['async_blocking_fraction']['label']}]")
    print(f"  job MTBF {M:.1f} h from per-GPU MTBF {v['gpu_mtbf_h']:,.0f} h [{inp['gpu_mtbf_h']['label']}]; restart "
          f"{v['restart_min']} min [{inp['restart_min']['label']}]")
    print(f"  checkpoint every {tau * 60:.0f} min (numerical optimum; Young {G.young(delta / 3600, M) * 60:.0f} min); "
          f"goodput {gp:.1%}")
    print(f"  wall clock {wall_days:.0f} days on {world} GPUs; cost {gpu_h / gp * v['usd_per_gpu_h'] / 1e6:.1f}M USD "
          f"[{inp['usd_per_gpu_h']['label']}]   (ALL PROJECTED)")


if __name__ == "__main__":
    main()
