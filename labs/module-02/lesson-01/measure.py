"""Predict, then measure: the roofline of this machine and one training step (lab 02.1).

    python labs/module-02/lesson-01/measure.py                               # free CPU: toy model
    python labs/module-02/lesson-01/measure.py --device cuda --dtype bf16 \\
        --preset baseline0 --batch 8 --seq 1024 --hw H100-SXM                 # main path
    LAB_TARGET=solution python labs/module-02/lesson-01/measure.py           # with the reference

Part 1 measures this machine's empirical roofline: a large GEMM (compute ceiling) and a large copy
(bandwidth ceiling). Part 2 times four ops and prints, for each, the intensity your cost functions
give, the time the roofline predicts and the time measured. Part 3 predicts one training step of
the chosen model *before* timing it, then times it and prints the implied MFU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from frontierlab.labkit import load_target
from frontierlab.layers.rmsnorm import RMSNorm
from frontierlab.model import LM, PRESETS
from frontierlab.perf.roofline import HARDWARE
from frontierlab.perf.timing import benchmark

lab = load_target(str(Path(__file__).parent / "test_lab.py"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--preset", default="toy", choices=sorted(PRESETS))
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--hw", default=None, help=f"datasheet entry to compare with: {sorted(HARDWARE)}")
    ap.add_argument("--repeats", type=int, default=10)
    a = ap.parse_args()
    dev = torch.device(a.device)
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    bpe = 2 if a.dtype == "bf16" else 4
    bench = lambda fn: benchmark(fn, warmup=3, repeats=a.repeats, device=dev)   # noqa: E731
    print(f"device {dev}, dtype {a.dtype}, torch {torch.__version__}, threads {torch.get_num_threads()}")

    # ---- 1. empirical roofline -------------------------------------------------------------
    # The compute ceiling is the best of a few large GEMM shapes (one shape can hit a slow path).
    shapes = ((4096, 4096, 4096), (8192, 2816, 768)) if dev.type == "cuda" else ((1024, 1024, 1024),
                                                                                  (2048, 2816, 768))
    peak = 0.0
    for m_, n_, k_ in shapes:
        A, Bm = torch.randn(m_, k_, device=dev, dtype=dt), torch.randn(k_, n_, device=dev, dtype=dt)
        peak = max(peak, 2 * m_ * n_ * k_ / bench(lambda: A @ Bm).median)
    src = torch.empty(64 * 2**20 // bpe * 4, device=dev, dtype=dt)          # 256 MiB of data
    dst = torch.empty_like(src)
    t = bench(lambda: dst.copy_(src))
    bw = 2 * src.numel() * bpe / t.median                                   # read src + write dst
    print(f"\n1. empirical roofline: best GEMM {peak / 1e12:.3f} TFLOP/s, copy {bw / 1e9:.1f} GB/s, "
          f"ridge {peak / bw:.1f} FLOP/byte")
    if a.hw:
        hw = HARDWARE[a.hw]
        print(f"   datasheet {hw.name}: {hw.peak_flops / 1e12:.0f} TFLOP/s dense {hw.dtype}, "
              f"{hw.mem_bw / 1e9:.0f} GB/s -> measured {peak / hw.peak_flops:.0%} and {bw / hw.mem_bw:.0%} of it")

    # ---- 2. four ops: predicted vs measured -----------------------------------------------------
    M = 8192 if dev.type == "cuda" else 2048
    C, I = 768, 2816
    x = torch.randn(M, C, device=dev, dtype=dt)
    w = torch.randn(C, I, device=dev, dtype=dt)
    x1 = x[:1].contiguous()
    norm = RMSNorm(C).to(dev, dt)
    s = torch.randn(M, 1024, device=dev, dtype=dt)
    kv_S, H, KV, d = 8192, 12, 4, 64
    q = torch.randn(16, H, 1, d, device=dev, dtype=dt)
    k = torch.randn(16, KV, kv_S, d, device=dev, dtype=dt)
    v = torch.randn_like(k)
    ops = [
        ("GEMM (M x 768) @ (768 x 2816)", lab.gemm_cost(M, I, C, bpe), lambda: x @ w),
        ("same GEMM, 1 token (decode)", lab.gemm_cost(1, I, C, bpe), lambda: x1 @ w),
        ("RMSNorm (M x 768)", lab.rmsnorm_cost(M, C, bpe), lambda: norm(x)),
        ("softmax (M x 1024)", lab.softmax_cost(M, 1024, bpe), lambda: torch.softmax(s, -1)),
        ("decode attention B=16 S=8192", lab.decode_attention_cost(16, H, KV, kv_S, d, bpe),
         lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True)),
    ]
    print(f"\n2. {'op':34s} {'FLOP/B':>8s} {'bound':>8s} {'roofline':>10s} {'measured':>10s} {'ratio':>6s}")
    for name, (f, b), fn in ops:
        pred = lab.roofline_time(f, b, peak, bw)
        meas = bench(fn).median
        bound = "compute" if f / b >= peak / bw else "memory"
        print(f"   {name:34s} {f / b:8.2f} {bound:>8s} {pred * 1e6:8.1f}us {meas * 1e6:8.1f}us {meas / pred:6.1f}")

    # ---- 3. predict, then measure, one training step ----------------------------------------------
    cfg = PRESETS[a.preset](vocab_size=a.vocab)
    pred_lb = lab.predict_step_time(cfg, a.batch, a.seq, peak, mfu=1.0)
    print(f"\n3. {a.preset}: predicted step time at MFU 1.0 (vs this machine's GEMM peak) {pred_lb * 1e3:.1f} ms;"
          f" at MFU 0.4 {pred_lb / 0.4 * 1e3:.1f} ms")
    torch.manual_seed(0)
    model = LM(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=dev.type == "cuda")
    idx = torch.randint(0, a.vocab, (a.batch, a.seq), device=dev)
    use_ac = a.dtype == "bf16"

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_ac):
            loss = model(idx, labels=idx).loss
        loss.backward()
        opt.step()

    t = bench(step)
    print(f"   measured: {t}")
    print(f"   implied MFU vs this machine's GEMM peak: {pred_lb / t.median:.1%}")
    if a.hw:
        print(f"   implied MFU vs {a.hw} datasheet peak: "
              f"{pred_lb * peak / HARDWARE[a.hw].peak_flops / t.median:.1%}")


if __name__ == "__main__":
    main()
