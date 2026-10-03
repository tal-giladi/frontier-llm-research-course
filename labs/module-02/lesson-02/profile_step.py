"""Profile one training step, then measure its activation memory four ways (lab 02.2).

    python labs/module-02/lesson-02/profile_step.py                         # free CPU, toy model
    python labs/module-02/lesson-02/profile_step.py --device cuda --dtype bf16 --preset baseline0 \\
        --vocab 32768 --batch 8 --seq 1024 --trace runs/l22/trace.json --snapshot runs/l22/mem.pickle
    python labs/module-02/lesson-02/profile_step.py --device cuda ... --compile     # compare torch.compile

Part 1: ``torch.profiler`` summary of a full step (forward, backward, AdamW) and the time shares per
category from your ``categorize``. Part 2: per-op dispatch overhead and what it costs per step.
Part 3: activation bytes saved for backward with the plain loss, checkpointed blocks, the chunked
loss, and both; plus the step time of each (checkpointing buys memory with recomputation).
On CUDA, ``--snapshot`` writes a memory snapshot (open it at https://pytorch.org/memory_viz) and
``--trace`` a timeline (open it at https://ui.perfetto.dev).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from frontierlab.labkit import load_target
from frontierlab.model import LM, PRESETS
from frontierlab.perf.chunked_ce import lm_loss_chunked
from frontierlab.perf.memory import logits_bytes, memory_snapshot, saved_activation_bytes
from frontierlab.perf.profiling import format_summary, profile_steps, summarize
from frontierlab.perf.timing import benchmark, interleaved

lab = load_target(str(Path(__file__).parent / "test_lab.py"))


def ckpt_and_chunked(model, idx, chunk):
    """Checkpointed blocks *and* the chunked loss (both savings at once)."""
    from torch.utils.checkpoint import checkpoint

    from frontierlab.perf.chunked_ce import chunked_cross_entropy
    T = idx.shape[1]
    pos = torch.arange(T, device=idx.device)
    x = model.model.embed_tokens(idx)
    for layer in model.model.layers:
        x = checkpoint(layer, x, pos, None, use_reentrant=False)
    h = model.model.norm(x)[:, :-1]
    return chunked_cross_entropy(h.reshape(-1, h.shape[-1]), model.lm_head.weight, idx[:, 1:].reshape(-1), chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--preset", default="toy", choices=sorted(PRESETS))
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--trace", default=None)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--compile", action="store_true", help="also time the step under torch.compile")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.manual_seed(0)
    cfg = PRESETS[a.preset](vocab_size=a.vocab)
    model = LM(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=dev.type == "cuda")
    idx = torch.randint(0, a.vocab, (a.batch, a.seq), device=dev)
    ac = a.dtype == "bf16"

    def make_step(loss_fn):
        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=ac):
                loss = loss_fn()
            loss.backward()
            opt.step()
        return step

    plain = lambda: model(idx, labels=idx).loss                     # noqa: E731
    print(f"{a.preset} B={a.batch} T={a.seq} V={a.vocab} on {dev} ({a.dtype}), torch {torch.__version__}")

    # ---- 1. profile ------------------------------------------------------------------------
    for d in (a.trace, a.snapshot):
        if d:
            Path(d).parent.mkdir(parents=True, exist_ok=True)
    prof = profile_steps(make_step(plain), steps=3, warmup=2, device=a.device, trace_path=a.trace)
    s = summarize(prof, top=15)
    print("\n1. profiler, top operators by self " + ("GPU" if dev.type == "cuda" else "CPU") + " time")
    print(format_summary(s))
    print("   by category (top 15 rows):", {k: f"{v:.0%}" for k, v in lab.categorize(s["top"]).items()})

    # ---- 2. dispatch overhead ------------------------------------------------------------------
    tiny = torch.zeros(1, device=dev)
    t = benchmark(lambda: [tiny.add_(1) for _ in range(1000)], warmup=2, repeats=5, device=dev)
    per_op = t.median / 1000
    print(f"\n2. one tiny op costs {per_op * 1e6:.1f} us end to end; {s['n_ops']} aten ops/step -> "
          f"{per_op * s['n_ops'] * 1e3:.1f} ms of pure overhead ({per_op * s['n_ops'] / (s['step_ms'] / 1e3):.0%} of "
          "the profiled step)")

    # ---- 3. activation memory and the cost of saving it -------------------------------------------
    variants = {
        "plain loss": plain,
        "checkpointed blocks": lambda: lab.checkpointed_loss(model, idx),
        "chunked loss": lambda: lm_loss_chunked(model, idx, a.chunk),
        "checkpointed + chunked": lambda: ckpt_and_chunked(model, idx, a.chunk),
    }
    lg = logits_bytes(a.batch, a.seq - 1, a.vocab)
    print(f"\n3. activation bytes saved for backward (fp32 logits alone: {lg / 2**20:.1f} MiB)")
    saved = {}
    for name, fn in variants.items():
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=ac):
            saved[name] = saved_activation_bytes(model, fn).total_bytes
    # interleaved rounds, so background load and drift hit every variant equally (lesson 02.4)
    times = interleaved({n: make_step(fn) for n, fn in variants.items()}, warmup=2, rounds=5, device=dev)
    for name in variants:
        lo, hi = times[name].ci()
        print(f"   {name:24s} {saved[name] / 2**20:9.1f} MiB   step {times[name].median * 1e3:8.1f} ms"
              f"  [{lo * 1e3:.1f}, {hi * 1e3:.1f}]")

    if a.snapshot:
        with memory_snapshot(a.snapshot):
            for _ in range(2):
                make_step(plain)()
        print(f"\n   memory snapshot written to {a.snapshot}; open it at https://pytorch.org/memory_viz")

    if a.compile:
        cmodel = torch.compile(model)
        step = make_step(lambda: cmodel(idx, labels=idx).loss)
        t0 = time.perf_counter()
        try:
            step()
        except Exception as e:          # noqa: BLE001 - e.g. no C++ compiler for Inductor on Windows CPU
            print(f"\n   torch.compile failed: {type(e).__name__}: {str(e)[:200]}")
            return
        print(f"\n   torch.compile first step (graph capture + code generation): {time.perf_counter() - t0:.1f} s")
        eager = benchmark(make_step(plain), warmup=2, repeats=10, device=dev)
        comp = benchmark(step, warmup=2, repeats=10, device=dev)
        print(f"   eager    {eager}\n   compiled {comp}")


if __name__ == "__main__":
    main()
