"""Lab 08.2, step 5: training-step throughput, BF16 vs torchao Float8 (real kernels), with the Module 2 harness.

    python labs/module-08/lesson-02/bench_fp8.py                                   # CPU: prints the PROJECTED bounds only
    python labs/module-08/lesson-02/bench_fp8.py --device cuda --hw L4 --preset pilot-30m --batch 8 --seq 512    # Colab L4 pilot
    python labs/module-08/lesson-02/bench_fp8.py --device cuda --hw H100-SXM --preset baseline0 --batch 16 --seq 1024   # main path

GPU path (not run in this build; part of the Module 8 pilot): three copies of one initial model — BF16 (bf16
autocast), torchao ``tensorwise`` and torchao ``rowwise`` — each wrapped in ``torch.compile``; one optimizer step
per call on the same random batch; ``frontierlab.perf.interleaved`` (warm-up 5 rounds, which absorbs compilation,
then 30 interleaved rounds with ``torch.cuda.synchronize`` around every sample) and the paired speed-up with a 95%
bootstrap interval; tokens/s and MFU (BF16-peak convention, so the numbers are comparable across arms); peak
memory per arm. Before timing, a correctness gate: one step's loss of each FP8 arm must be within 2% of BF16's
(an FP8 arm that diverged or silently fell back is not timed).

Every run also prints the roofline projection of ``frontierlab.precision.cost`` so the measured number can be put
next to its bound.
"""

import argparse
import copy

import torch

from frontierlab.flops import PEAK_BF16, flops_per_token
from frontierlab.model import LM, PRESETS
from frontierlab.precision.cost import projected_speedup


def projection(cfg, B, T, hw):
    for fused in (False, True):
        r = projected_speedup(cfg, B, T, hw, "fp8", fused)
        print(f"PROJECTED ({'casts fused' if fused else 'casts unfused'}): BF16 step {r['t_bf16'] * 1e3:.2f} ms, "
              f"FP8 {r['t_lowp'] * 1e3:.2f} ms, speed-up {r['speedup']:.3f}x; block linears are "
              f"{r['linear_share_bf16']:.0%} of the BF16 step (Amdahl limit {r['amdahl_limit']:.3f}x)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="pilot-30m")
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--hw", default="H100-SXM", choices=["H100-SXM", "L4", "A100-SXM-80GB"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rounds", type=int, default=30)
    a = ap.parse_args(argv)
    cfg = PRESETS[a.preset](vocab_size=a.vocab)
    print(f"{a.preset}, batch {a.batch} x {a.seq}, vocab {a.vocab}, roofline on {a.hw}")
    projection(cfg, a.batch, a.seq, a.hw)
    if not a.device.startswith("cuda"):
        print("\nCPU: no FP8 kernels here; emulation measures numerics, not speed. Run this on an L4 or H100.")
        return None
    from frontierlab.perf.timing import interleaved, speedup
    from frontierlab.precision.torchao_path import convert_torchao_float8, fp8_capable
    if not fp8_capable():
        raise SystemExit("this GPU has no FP8 tensor cores (needs sm89+: L4, H100, ...)")
    torch.manual_seed(0)
    base = LM(cfg)
    arms = {}
    for name in ("bf16", "tensorwise", "rowwise"):
        m = copy.deepcopy(base)
        if name != "bf16":
            convert_torchao_float8(m, name)
        m = m.cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
        arms[name] = (m, torch.compile(m), opt)
    x = torch.randint(0, a.vocab, (a.batch, a.seq), device="cuda")

    def step_fn(name):
        m, f, opt = arms[name]

        def fn():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16):
                loss = f(x, labels=x).loss
            loss.backward()
            opt.step()
            return loss
        return fn

    fns = {n: step_fn(n) for n in arms}
    first = {n: float(fns[n]()) for n in fns}
    print("\ncorrectness gate, first-step loss:", {n: round(v, 4) for n, v in first.items()})
    for n in ("tensorwise", "rowwise"):
        if abs(first[n] - first["bf16"]) > 0.02 * first["bf16"]:
            raise SystemExit(f"{n}: first-step loss differs from BF16 by more than 2%; not timing it")
    torch.cuda.reset_peak_memory_stats()
    res = interleaved(fns, warmup=5, rounds=a.rounds, device="cuda")
    tok = a.batch * a.seq
    fpt = flops_per_token(cfg, a.seq)
    for n, t in res.items():
        print(f"{n:11s} {t}  {tok / t.median:,.0f} tok/s  MFU(bf16 peak) {fpt * tok / t.median / PEAK_BF16[a.hw]:.3f}")
    for n in ("tensorwise", "rowwise"):
        s = speedup(res["bf16"], res[n])
        print(f"speed-up {n} over bf16: {s['speedup']:.3f}x, 95% CI [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]  (measured)")
    print(f"peak memory over all arms: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return res


if __name__ == "__main__":
    main()
