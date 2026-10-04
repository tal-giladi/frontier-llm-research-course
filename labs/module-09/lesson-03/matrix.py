"""Build the tested-capability matrix for the course model (lesson 09.3).

    python labs/module-09/lesson-03/matrix.py                                  # CPU probes on 4 gloo processes
    python labs/module-09/lesson-03/matrix.py --gpu-results gpu_matrix.json    # main path: add your GPU column

Three columns per feature: what torchtitan 0.3.0 *documents* (its README at tag v0.3.0, read 2026-10-04), what the
course's CPU probes found for the course model on this machine (``python -m frontierlab.dist.capability``), and —
on the main path — what your torchtitan runs on the 8-GPU node found (a JSON file {feature: "composed" | "failed:
<reason>" | "not tried"} that you write after run_titan.sh). Writes ``runs/m09/capability_matrix.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from frontierlab.dist import capability

# torchtitan README at tag v0.3.0 ("Key features"), as listed there; checked 2026-10-04
DOCUMENTED = {
    "FSDP2": "FSDP2 with per-parameter sharding",
    "TP": "Tensor Parallel (including async TP)",
    "PP": "Pipeline Parallel",
    "CP": "Context Parallel",
    "Activation checkpointing": "Per-op selective and full activation checkpointing",
    "DCP": "Distributed checkpointing (including async checkpointing)",
    "Float8": "Float8 support",
    "MXFP8": "MXFP8 training for dense and MoE models",
    "torch.compile": "torch.compile support",
}

CPU_ROW = {  # which CPU probe gives evidence for each documented feature
    "FSDP2": ["FSDP2"],
    "TP": ["TP (DTensor), course attention unchanged", "TP (DTensor) + per-rank head counts",
           "TP + per-rank heads + QK-norm grad all-reduce", "FSDP2 x TP (2-D)"],
    "PP": ["PP (torch.distributed.pipelining, 1F1B)"],
    "CP": ["CP: ring attention (course code)"],
    "Activation checkpointing": ["FSDP2 + activation checkpointing"],
    "DCP": ["DCP save FSDP2 -> load unsharded", "DCP async_save under FSDP2"],
    "Float8": ["Float8 linears (torchao)"],
    "MXFP8": ["MXFP8 / NVFP4 linears"],
    "torch.compile": ["torch.compile + FSDP2"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--gpu-results", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/m09/capability_matrix.md"))
    a = ap.parse_args()
    rows = capability.run_matrix(a.world)
    env = capability.environment()
    by_name = {r["feature"]: r for r in rows}
    gpu = json.loads(a.gpu_results.read_text()) if a.gpu_results else {}
    lines = ["# Tested-capability matrix: course model (GQA + QK-norm + RoPE + SwiGLU, tied embeddings)", "",
             f"CPU probes: torch {env['torch']}, {env['platform']}, {a.world} gloo processes, float64.", "",
             "| Feature | Documented by torchtitan 0.3.0 | CPU probe (this machine) | Main path: torchtitan on 8 GPUs |",
             "|---|---|---|---|"]
    for feat, doc in DOCUMENTED.items():
        cpu = "; ".join(f"{n}: **{by_name[n]['status']}**" for n in CPU_ROW[feat] if n in by_name)
        lines.append(f"| {feat} | {doc} | {cpu} | {gpu.get(feat, 'not run in this build')} |")
    lines += ["", "## Probe evidence", "", capability.to_markdown(rows, env)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
