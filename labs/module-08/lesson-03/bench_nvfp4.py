"""Lab 08.3, main path only: NVFP4 vs MXFP8 vs BF16 GEMM throughput on a B200 with NVIDIA Transformer Engine.

    python labs/module-08/lesson-03/bench_nvfp4.py                       # anywhere: prints the PROJECTED bounds
    python labs/module-08/lesson-03/bench_nvfp4.py --device cuda         # 1x B200 (sm100): NOT PILOTED

NOT PILOTED: Colab has no Blackwell GPU, so this script was not run in this build or in the Module 8 pilot. Transformer
Engine is not pinned in references/versions.md (proposed in curriculum/inbox/module-08-shared-changes.md); the API
used below — ``transformer_engine.pytorch.Linear``, ``te.fp8_autocast(enabled=True, fp8_recipe=...)`` and the recipes
``NVFP4BlockScaling``, ``MXFP8BlockScaling`` from ``transformer_engine.common.recipe`` — is the one in TE's
documentation (release 2.18) and must be re-checked against the installed version before use. TE documents NVFP4
training on SM 10.0 and 10.3 only.

What it measures: forward + backward of a stack of ``te.Linear`` layers with Baseline-0's linear shapes (one block:
q, k, v, o, gate, up, down) at a representative token count, BF16 vs MXFP8 vs NVFP4, timed with
``frontierlab.perf.interleaved`` (warm-up, synchronisation, 30 interleaved rounds) and paired speed-ups.
"""

import argparse

import torch

from frontierlab.model import PRESETS
from frontierlab.precision.cost import projected_speedup


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="baseline0")
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rounds", type=int, default=30)
    a = ap.parse_args(argv)
    cfg = PRESETS[a.preset](vocab_size=32768)
    for fused in (False, True):
        r = projected_speedup(cfg, a.tokens // 1024, 1024, "B200", "fp4", fused)
        print(f"PROJECTED B200 FP4 linears ({'fused' if fused else 'unfused'} casts): whole step {r['speedup']:.3f}x; "
              f"linears alone {r['t_linear_bf16'] / r['t_linear_lowp']:.2f}x; block linears {r['linear_share_bf16']:.0%} "
              f"of the BF16 step (Amdahl limit {r['amdahl_limit']:.3f}x)")
    if not a.device.startswith("cuda"):
        print("\nNo Blackwell GPU here: nothing measured.")
        return None
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import MXFP8BlockScaling, NVFP4BlockScaling

    from frontierlab.perf.timing import interleaved, speedup
    C, I, H, KV, d = cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    shapes = [(C, H * d), (C, KV * d), (C, KV * d), (H * d, C), (C, I), (C, I), (I, C)]
    layers = [te.Linear(k, n, bias=False, params_dtype=torch.bfloat16).cuda() for k, n in shapes]
    xs = [torch.randn(a.tokens, k, device="cuda", dtype=torch.bfloat16, requires_grad=True) for k, _ in shapes]
    recipes = {"bf16": None, "mxfp8": MXFP8BlockScaling(), "nvfp4": NVFP4BlockScaling()}

    def make(rec):
        def fn():
            with te.fp8_autocast(enabled=rec is not None, fp8_recipe=rec):
                outs = [lin(x) for lin, x in zip(layers, xs)]
            sum(o.float().sum() for o in outs).backward()
        return fn

    res = interleaved({k: make(v) for k, v in recipes.items()}, warmup=5, rounds=a.rounds, device="cuda")
    flops = sum(6 * a.tokens * k * n for k, n in shapes)
    for k, t in res.items():
        print(f"{k:6s} {t}  {flops / t.median / 1e12:,.0f} TFLOP/s (fwd + bwd of one block's linears)")
    for k in ("mxfp8", "nvfp4"):
        s = speedup(res["bf16"], res[k])
        print(f"speed-up {k} over bf16: {s['speedup']:.3f}x, 95% CI [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]  (measured)")
    return res


if __name__ == "__main__":
    main()
