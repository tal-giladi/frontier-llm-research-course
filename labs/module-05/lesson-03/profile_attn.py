"""Lab 05.3: component costs of dense, linear (KDA) and DSA-style attention across contexts.

    python labs/module-05/lesson-03/profile_attn.py                                   # free CPU, pilot-10m layer shape
    python labs/module-05/lesson-03/profile_attn.py --mode decode
    python labs/module-05/lesson-03/profile_attn.py --device cuda --dtype bf16 --shape baseline0 \\
        --contexts 8192 16384 32768 65536 131072 --topk 2048 --chunk 64 --linear-mode fla   # main path

For every context: each component on its own (median ms, 95% CI), FLOPs and minimum bytes, then the three
whole-layer arms in interleaved rounds with paired speed-ups against dense and op / kernel counts. At the
end: the fitted growth exponent of each arm (t ~ a L^p; 2 = quadratic, 1 = linear), the measured crossover
contexts (your ``crossover``), and the roofline projection of every component to an H100 (your
``project_time``), labelled PROJECTED.

Writes everything to ``--out`` (JSON) for the project memo.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from frontierlab.attention import subq_bench as sb
from frontierlab.labkit import load_target
from frontierlab.model import PRESETS
from frontierlab.perf.roofline import HARDWARE

lab = load_target(str(Path(__file__).parent / "test_lab.py"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", default="pilot-10m", choices=sorted(PRESETS), help="layer shape taken from this preset")
    ap.add_argument("--contexts", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    ap.add_argument("--mode", choices=["prefill", "decode"], default="prefill")
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--index-heads", type=int, default=4)
    ap.add_argument("--index-dim", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--block", type=int, default=512, help="query block for indexer / top-k / gather")
    ap.add_argument("--linear-mode", default=None, choices=["chunked", "recurrent", "fla"])
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--project-to", default="H100-SXM", choices=sorted(HARDWARE))
    ap.add_argument("--mfu", type=float, default=0.4, help="assumed fraction of peak for compute-bound components")
    ap.add_argument("--bw-eff", type=float, default=0.7, help="assumed fraction of peak bandwidth")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    print(f"checking {os.environ.get('LAB_TARGET', 'lab')}.py   torch {torch.__version__}  threads {torch.get_num_threads()}")
    cfg = PRESETS[a.shape]()
    sh = sb.Shape.of(cfg, k=a.topk, chunk=a.chunk, HI=a.index_heads, dI=a.index_dim)
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    print(f"layer shape {a.shape}: H={sh.H} KV={sh.KV} d={sh.d}; indexer H_I={sh.HI} d_I={sh.dI}; k={sh.k}; "
          f"chunk={sh.chunk}; {a.mode}; {a.device} {a.dtype}\n")
    rows = []
    for L in a.contexts:
        r = sb.measure(sh, L, decode=a.mode == "decode", device=a.device, dtype=dt, block=a.block,
                       repeats=a.repeats, rounds=a.rounds, linear_mode=a.linear_mode)
        rows.append(r)
        print(f"L = {L}")
        for name, c in r["components"].items():
            print(f"   {name:<12s} {c['ms']:10.3f} ms [{c['ci_ms'][0]:9.3f}, {c['ci_ms'][1]:9.3f}]  "
                  f"{c['flops'] / 1e9:9.3f} GFLOP  {c['bytes'] / 2**20:9.2f} MiB")
        for name, arm in r["arms"].items():
            s = f"   arm {name:<8s} {arm['ms']:10.3f} ms [{arm['ci_ms'][0]:9.3f}, {arm['ci_ms'][1]:9.3f}]  ops {arm.get('ops', '-')}"
            if "speedup_vs_dense" in arm:
                sp = arm["speedup_vs_dense"]
                s += f"  speed-up vs dense {sp['speedup']:.3f} [{sp['ci'][0]:.3f}, {sp['ci'][1]:.3f}]"
            print(s)
    Ls = [r["L"] for r in rows]
    print("\nGrowth exponents (t ~ a L^p, least squares in log-log over the measured contexts)")
    times = {arm: [r["arms"][arm]["ms"] for r in rows] for arm in ("dense", "linear", "dsa")}
    for arm, t in times.items():
        print(f"   {arm:<8s} p = {lab.fit_exponent(Ls, t):.2f}")
    for comp in rows[0]["components"]:
        print(f"   {comp:<12s} p = {lab.fit_exponent(Ls, [r['components'][comp]['ms'] for r in rows]):.2f}")
    print("\nMeasured crossovers (smallest context at which the arm is faster than dense; log-log interpolation)")
    for arm in ("linear", "dsa"):
        c = lab.crossover(Ls, times["dense"], times[arm])
        print(f"   {arm:<8s} {c if c is not None else 'not within the measured contexts'}")
    hw = HARDWARE[a.project_to]
    print(f"\nPROJECTED on {hw.name}: max(FLOPs / (peak x {a.mfu}), bytes / (bandwidth x {a.bw_eff})) per component,"
          f" BF16 bytes. Lower bounds for good kernels; launch overheads not included")
    proj = []
    for L in a.contexts:
        fb = sb.component_flops_bytes(sh, L, a.mode == "decode", b=2)
        t = {k: lab.project_time(f, b, hw.peak_flops, hw.mem_bw, a.mfu, a.bw_eff) * 1e3 for k, (f, b) in fb.items()}
        dsa = sum(v for k, v in t.items() if k.startswith("dsa."))
        lin = t.get("linear.core", t.get("linear.step"))
        proj.append({"L": L, **t, "dsa.total": dsa})
        print(f"   L={L:>7d}  dense {t['dense.sdpa']:9.4f} ms   linear {lin:9.4f} ms   dsa {dsa:9.4f} ms "
              f"(indexer {t['dsa.indexer']:.4f}, topk {t['dsa.topk']:.4f}, gather {t['dsa.gather']:.4f}, attend {t['dsa.attend']:.4f})")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"args": vars(a), "shape": sh.__dict__, "rows": rows, "projected": proj},
                                          default=float))
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
