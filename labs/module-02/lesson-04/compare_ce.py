"""End-to-end, equal-work comparison: Baseline-0's plain loss vs the chunked cross-entropy (lab 02.4).

    python labs/module-02/lesson-04/compare_ce.py                                   # free CPU, toy model
    python labs/module-02/lesson-04/compare_ce.py --device cuda --dtype bf16 --preset baseline0 \\
        --vocab 32768 --batch 8 --seq 1024 --steps 40 --max-batch                     # main path

Two copies of the same initial model train side by side on the *same* batches (one data generator,
each batch used by both arms) with AdamW. Each step is timed for arm A then arm B on that batch
(interleaved), so drift and background load hit both. The script reports:

1. correctness: loss and gradient agreement on one batch (your ``grad_agreement``), and the largest
   difference between the two arms' training losses over all steps;
2. memory: saved activation bytes per arm (and, on CUDA, peak allocated memory);
3. time: median step time per arm with intervals, and the paired speed-up with its interval;
4. the decision from your ``decide``, using the rule in the lesson's experiment contract;
5. with ``--max-batch`` on CUDA: the largest micro-batch that fits per arm (doubling until out of
   memory) and the tokens/s each reaches there — the end-to-end benefit of the memory saving.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from frontierlab.labkit import load_target
from frontierlab.model import LM, PRESETS
from frontierlab.perf.chunked_ce import lm_loss_chunked
from frontierlab.perf.memory import cuda_peak_bytes, saved_activation_bytes
from frontierlab.perf.timing import Timing, speedup, sync

lab = load_target(str(Path(__file__).parent / "test_lab.py"))


def plain_loss(model, idx):
    return model(idx, labels=idx).loss


def batches(a, dev):
    """Data-v0 training windows if prepared at this vocab size, otherwise uniform random tokens."""
    g = torch.Generator().manual_seed(a.seed)
    try:
        from frontierlab.data.loader import TokenData
        data = TokenData("train")
        if data.meta["vocab_size"] != a.vocab:
            raise ValueError("vocab mismatch")
        print(f"data: Data-v0 train split ({len(data):,} tokens)")
        return lambda B: data.batch(B, a.seq, g, dev)
    except (FileNotFoundError, ValueError) as e:
        print(f"data: random tokens ({e})")
        return lambda B: torch.randint(0, a.vocab, (B, a.seq), generator=g).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--preset", default="toy", choices=sorted(PRESETS))
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-batch", action="store_true", help="CUDA only: search the largest micro-batch per arm")
    ap.add_argument("--out", default=None, help="write the results as JSON here")
    a = ap.parse_args()
    dev = torch.device(a.device)
    ac = a.dtype == "bf16"
    torch.manual_seed(a.seed)
    cfg = PRESETS[a.preset](vocab_size=a.vocab)
    base = LM(cfg).to(dev)
    arms = {"plain": (base, plain_loss),
            "chunked": (copy.deepcopy(base), lambda m, i: lm_loss_chunked(m, i, a.chunk))}
    opts = {k: torch.optim.AdamW(m.parameters(), lr=a.lr, fused=dev.type == "cuda") for k, (m, _) in arms.items()}
    next_batch = batches(a, dev)
    print(f"{a.preset} B={a.batch} T={a.seq} V={a.vocab} {a.dtype} on {dev}, torch {torch.__version__}")

    # 1. correctness on one batch, in float64 on CPU (a copy, so training is unaffected)
    x0 = next_batch(a.batch)[:2, :64].cpu()
    m64 = copy.deepcopy(base).cpu().double()
    ga = lab.grad_agreement(m64, x0, plain_loss, lambda m, i: lm_loss_chunked(m, i, 16))
    correct = ga["loss_diff"] < 1e-6 and ga["max_grad_diff"] < 1e-6
    print(f"\n1. float64 agreement on one batch: |loss diff| {ga['loss_diff']:.1e}, max |grad diff| "
          f"{ga['max_grad_diff']:.1e} -> {'PASS' if correct else 'FAIL'}")

    def step(name, x):
        model, loss_fn = arms[name]
        opts[name].zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=ac):
            loss = loss_fn(model, x)
        loss.backward()
        opts[name].step()
        return loss.detach()

    # 2. memory
    x = next_batch(a.batch)
    mem = {}
    for name, (model, loss_fn) in arms.items():
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=ac):
            mem[name] = saved_activation_bytes(model, lambda: loss_fn(model, x)).total_bytes
    peak = {}
    if dev.type == "cuda":
        for name in arms:
            peak[name] = cuda_peak_bytes(lambda: step(name, x))
    mem_ratio = mem["chunked"] / mem["plain"]
    print(f"\n2. saved activations: plain {mem['plain'] / 2**20:.1f} MiB, chunked {mem['chunked'] / 2**20:.1f} MiB "
          f"(ratio {mem_ratio:.2f})" + (f"; CUDA peak {peak['plain'] / 2**30:.2f} vs {peak['chunked'] / 2**30:.2f} GiB"
                                        if peak else ""))

    # 3. equal-work training: same batches, interleaved timing
    times = {k: Timing() for k in arms}
    losses = {k: [] for k in arms}
    import time
    for s in range(a.warmup + a.steps):
        x = next_batch(a.batch)
        for name in arms:
            sync(dev)
            t0 = time.perf_counter()
            loss = step(name, x)
            sync(dev)
            dt = time.perf_counter() - t0
            if s >= a.warmup:
                times[name].samples.append(dt)
                losses[name].append(loss.item())
            else:
                times[name].warmup.append(dt)
    drift = max(abs(p - c) for p, c in zip(losses["plain"], losses["chunked"]))
    sp = speedup(times["plain"], times["chunked"])
    tok = a.batch * a.seq
    print(f"\n3. {a.steps} timed steps after {a.warmup} warm-up steps, same batches for both arms")
    for name in arms:
        print(f"   {name:8s} {times[name]}   {tok / times[name].median:,.0f} tok/s   final loss {losses[name][-1]:.4f}")
    print(f"   speed-up chunked/plain {sp['speedup']:.3f}  95% CI [{sp['ci'][0]:.3f}, {sp['ci'][1]:.3f}]")
    print(f"   largest training-loss difference between arms over {a.steps} steps: {drift:.2e}")

    decision = lab.decide(correct, sp["ci"], mem_ratio)
    print(f"\n4. decision (rule from the contract): {decision.upper()}")

    result = {"preset": a.preset, "batch": a.batch, "seq": a.seq, "vocab": a.vocab, "dtype": a.dtype,
              "device": str(dev), "correct": correct, **ga, "saved_bytes": mem, "cuda_peak": peak,
              "median_s": {k: times[k].median for k in arms}, "ci_s": {k: times[k].ci() for k in arms},
              "speedup": sp, "loss_drift": drift, "decision": decision}

    if a.max_batch and dev.type == "cuda":
        print("\n5. largest micro-batch that fits, and throughput there")
        result["max_batch"] = {}
        for name in arms:
            B, best = a.batch, None
            while True:
                try:
                    xb = next_batch(B)
                    for _ in range(2):
                        step(name, xb)
                    sync(dev)
                    t0 = time.perf_counter()
                    for _ in range(5):
                        step(name, xb)
                    sync(dev)
                    best = (B, 5 * B * a.seq / (time.perf_counter() - t0))
                    B *= 2
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    break
            print(f"   {name:8s} max micro-batch {best[0]}  {best[1]:,.0f} tok/s")
            result["max_batch"][name] = best

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
