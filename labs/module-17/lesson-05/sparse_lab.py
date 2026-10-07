"""Lab 17.5 (extension): weight-sparse transformers have smaller circuits — and an introspection experiment with
its controls.

    python labs/module-17/lesson-05/sparse_lab.py            # part A, free CPU: about 3 minutes
    python labs/module-17/lesson-05/sparse_lab.py --hf       # adds part B on Qwen3-0.6B (~2-3 min CPU)
    python labs/module-17/lesson-05/sparse_lab.py --variant main --print

A. The closing-quote task (does a string opened with ' or " close with the same quote?). Dense models and models
   with 5% of each block weight matrix kept (annealed over the first half of training), two seeds each; every
   model pruned to the fewest nodes (MLP neurons and attention-output channels) that keep the task loss below
   0.15 with the rest mean-ablated; circuit size in nodes and in nonzero weights touching kept nodes.
B. (--hf) Concept injection on Qwen3-0.6B: concept vectors at layer 18 of 28, injected from the answer onward,
   at two strengths; the same number of trials with no injection and with random vectors of the same norm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from frontierlab.interp import sparse as SPR
from frontierlab.labkit import load_path

HERE = Path(__file__).resolve().parent

MAIN = """# Main path (1x GPU of any size; not run in this build, part of the Module 17 pilot).
# A at larger scale: width 256, 4 layers, 3 densities (1.0, 0.05, 0.01), 3 seeds, longer training:
python labs/module-17/lesson-05/sparse_lab.py --width 256 --layers 4 --densities 1.0 0.05 0.01 --seeds 3 --steps 6000
# B on the post-trained Stage D-size model (Qwen/Qwen3-1.7B, 70d244cc), more words and layers:
python labs/module-17/lesson-05/sparse_lab.py --hf --model qwen3-1.7b --layers-inject 14 18 22 --alphas 2 4 8 --device cuda
# PROJECTED (pending pilot): A ~9 runs x 6,000 steps of a 2M-parameter model; the course code trains it on the CPU
# (tens of minutes per run at width 256); B on the GPU:
# B 50 words x (1 + 3 layers x 3 strengths x 2) trials x 40 tokens ~ 38,000 decode steps, ~20-40 min (< 1 GPU-hour).
"""


def part_a(lab, densities, seeds, steps, width, layers):
    print("A. Closing-quote task: circuit size at matched loss (target 0.15 nats)")
    rows = []
    try:
        W = torch.randn(8, 8)
        print(f"  your magnitude mask keeps {int(lab.magnitude_mask(W, 0.25).sum())} of 64 entries at 0.25 (expected 16)")
        print(f"  your density schedule at step 25 of 100 (target 0.1): {lab.density_at(25, 100, 0.1):.3f} (expected 0.550)")
    except NotImplementedError as e:
        print(f"  {e} - finish the TODOs (the script uses the course's code)")
    for d in densities:
        for s in range(seeds):
            t = time.time()
            cfg_kw = {"hidden_size": width, "num_hidden_layers": layers}
            m = SPR.train_task_model(d, steps=steps, seed=s, **cfg_kw)
            acc = SPR.accuracy(m)
            r = SPR.prune_circuit(m, target_loss=0.15, steps=400, seed=s)
            rows.append((d, s, SPR.nonzero_fraction(m), acc, r))
            print(f"  density {d:4.2f} seed {s}: nonzero {SPR.nonzero_fraction(m):.3f}, accuracy {acc:.3f}; circuit "
                  f"{r['n_nodes']:3d} of {r['total_nodes']} nodes, {r['edges']:5d} nonzero weights, pruned loss "
                  f"{r['loss']:.3f}, accuracy {r['acc']:.3f} ({time.time() - t:.0f} s)")
    for d in densities:
        e = [r["edges"] for dd, _, _, _, r in rows if dd == d]
        n = [r["n_nodes"] for dd, _, _, _, r in rows if dd == d]
        print(f"  density {d:4.2f}: nodes {np.mean(n):.1f} (seeds {n}), edges {np.mean(e):.0f} (seeds {e}), "
              f"geometric-mean edges {np.exp(np.mean(np.log(e))):.0f}")
    return rows


def part_b(model_name, layers, alphas, device):
    from frontierlab.interp import hf
    from frontierlab.interp import introspect as IN
    model, tok = hf.load(model_name, device=device)
    print(f"\nB. Concept injection on {hf.MODELS[model_name][0]} ({len(IN.WORDS)} concepts)")
    out = {}
    for L in layers:
        t = time.time()
        r = IN.experiment(model, tok, L, alphas)
        out[L] = r["summary"]
        for k, v in r["summary"].items():
            print(f"  layer {L} {k:12s} n={v['n']:2d}: claims detection {v['claims']:2d} "
                  f"[{v['claims_ci'][0]:.2f}, {v['claims_ci'][1]:.2f}], claims and names the concept {v['correct']:2d} "
                  f"[{v['correct_ci'][0]:.2f}, {v['correct_ci'][1]:.2f}]")
        w, txt, g = r["trials"][f"concept@{alphas[-1]}"][0]
        print(f"  example (concept '{w}', strength {alphas[-1]}): {txt[:160]!r}")
        w, txt, g = r["trials"]["none"][0]
        print(f"  example (no injection): {txt[:160]!r}  ({time.time() - t:.0f} s)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["cpu", "main"], default="cpu")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--hf", action="store_true")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--densities", type=float, nargs="+", default=[1.0, 0.05])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--model", default="qwen3-0.6b")
    ap.add_argument("--layers-inject", type=int, nargs="+", default=[18])
    ap.add_argument("--alphas", type=float, nargs="+", default=[2.0, 4.0])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.print or a.variant == "main":
        print(MAIN)
        return
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    res = {}
    if not a.skip_a:
        res["a"] = [(d, s, nz, acc, {k: v for k, v in r.items() if k != "kept"})
                    for d, s, nz, acc, r in part_a(lab, a.densities, a.seeds, a.steps, a.width, a.layers)]
    if a.hf:
        res["b"] = part_b(a.model, a.layers_inject, a.alphas, a.device)
    out = Path("runs/m17/l175.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    print(f"\ntotal {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
