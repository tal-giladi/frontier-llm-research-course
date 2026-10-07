"""Lab 15.3: what the Module 3–5 architecture choices cost at serving time, and in RL rollouts.

    python labs/module-15/lesson-03/serve_lab.py                   # free CPU, about 5 minutes
    python labs/module-15/lesson-03/serve_lab.py --part kv         # kv | roofline | measure | serve | rollout
    python labs/module-15/lesson-03/serve_lab.py --variant main --print

Parts:

* ``kv`` — KV bytes at 128K for every Module 3–5 design on the Stage D model's shape (your functions next to the
  course's), and for released models from their pinned configs; sequences that fit on one 80 GB GPU.
* ``roofline`` — arithmetic intensity of prefill and decode, decode step time and tokens/s against batch and
  context for the Stage D shape on an H100 (lower bounds, PROJECTED).
* ``measure`` — measured on this CPU: prefill vs decode time per token for toy models of four attention kinds,
  cache bytes against the formula, and decode-step time as the context grows.
* ``serve`` — the iteration-level simulator: prefill-first, chunked prefill and disaggregated serving on 2 GPUs,
  for a long-prompt workload and a long-answer (reasoning) workload, TTFT / TPOT / goodput against the SLOs.
* ``rollout`` — one RL step's rollouts (256 sequences) for each design at 4K, 16K and 32K generated tokens on one
  H100 shared with the trainer: batch that fits, waves, roofline time.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from frontierlab.labkit import load_path
from frontierlab.perf.roofline import HARDWARE
from frontierlab.ttc import serving as SV

HERE = Path(__file__).resolve().parent
GiB = 2 ** 30
S128 = 131072
RELEASED = ["Qwen__Qwen3-8B", "deepseek-ai__DeepSeek-V3", "moonshotai__Kimi-K2-Instruct", "openai__gpt-oss-120b",
            "unsloth__gemma-3-27b-pt", "Qwen__Qwen3-Next-80B-A3B-Instruct", "MiniMaxAI__MiniMax-M2"]


def part_kv(lab):
    hw80 = 80 * GiB
    print(f"KV cache per sequence at {S128:,} tokens, BF16, on the Stage D shape (28 layers, 2048 wide; design "
          "calculation)")
    print(f"  {'design':30s} {'GiB/seq':>8s} {'B/token':>9s} {'fit on 80 GB':>12s}  yours")
    for r in SV.kv_table(SV.stage_d_designs(), S128):
        yours = ""
        if r["design"] == "GQA (as released)":
            yours = f"{lab.kv_bytes_per_token(28, 'gqa', kv_heads=8, head_dim=128) * S128 / GiB:.2f}"
        elif r["design"] == "MLA d_c=4d":
            yours = f"{lab.kv_bytes_per_token(28, 'mla', d_c=512, d_r=64) * S128 / GiB:.2f}"
        elif r["design"].startswith("local/global"):
            yours = f"{lab.local_global_bytes(S128, 24, 4, 1024, 8, 128) / GiB:.2f}"
        elif r["design"] == "MHA":
            yours = f"{lab.kv_bytes_per_token(28, 'gqa', kv_heads=16, head_dim=128) * S128 / GiB:.2f}"
        print(f"  {r['design']:30s} {r['kv_gib_per_seq']:8.2f} {r['kv_bytes_per_token']:9.0f} {r['fit_at_S']:12d}  {yours}")
    print(f"\nThe same designs on Baseline-0 (~122M parameters, 12 layers, 768 wide), at {S128:,} tokens")
    for r in SV.kv_table(SV.baseline0_designs(), S128):
        print(f"  {r['design']:30s} {r['kv_gib_per_seq']:8.3f} GiB  {r['kv_bytes_per_token']:9.0f} B/token")
    print(f"\nReleased models (pinned config snapshots, BF16 cache, {S128:,} tokens)")
    for name in RELEASED:
        r = SV.released_kv(name, S128)
        fit = SV.capacity(hw80, 0, r["kv_bytes"])
        print(f"  {name:40s} {r['kv_bytes'] / GiB:7.2f} GiB/seq  {r['kv_bytes'] / S128:9.0f} B/token  layers {r['kinds']}"
              f"  (cache-only fit on 80 GB: {fit})")


def part_roofline(lab):
    hw = HARDWARE["H100-SXM"]
    cfg = SV.QWEN3_1_7B
    w = SV.weight_bytes(cfg)
    print(f"Stage D shape on {hw.name}: {SV.n_params(cfg) / 1e9:.2f}B parameters, {w / GiB:.2f} GiB of BF16 weights; "
          f"ridge {hw.ridge:.0f} FLOP/byte")
    for T in (128, 512, 2048, 8192):
        print(f"  prefill T={T:5d}: intensity {SV.intensity(cfg, 'prefill', T=T):7.0f} FLOP/byte, "
              f"lower bound {SV.prefill_time(cfg, T, hw) * 1e3:7.2f} ms ({T / SV.prefill_time(cfg, T, hw):,.0f} tok/s)")
    print("  decode step lower bound (ms) and tokens/s, by batch (rows) and context (columns); yours in brackets")
    ctxs = (4096, 32768, 131072)
    print("  " + " " * 8 + "".join(f"{f'S={s}':>28s}" for s in ctxs))
    for B in (1, 8, 32, 128):
        cells = []
        for S in ctxs:
            kv = SV.kv_bytes_seq(cfg, S)
            fits = SV.capacity(80 * GiB, w, kv) >= B
            t = SV.decode_step_time(cfg, B, S, hw)
            mine = lab.decode_step_time(w, kv, B, SV.decode_flops(cfg, S), hw.peak_flops, hw.mem_bw)
            cells.append(f"{t * 1e3:7.2f} [{mine * 1e3:6.2f}] {B / t:7.0f}" + (" " if fits else "*"))
        print(f"  B={B:4d}  " + "".join(f"{c:>28s}" for c in cells))
    print("  (* = does not fit in 80 GB at that context; intensity of a decode step at B=1, S=4096: "
          f"{SV.intensity(cfg, 'decode', B=1, S=4096):.1f} FLOP/byte)")
    for S in ctxs:
        th = SV.decode_throughput(cfg, S, hw)
        print(f"  largest batch at S={S:6d}: {th['batch']:4d} sequences, {th['tokens_per_s']:9,.0f} tokens/s "
              f"(capacity yours: {lab.capacity(80 * GiB, w, SV.kv_bytes_seq(cfg, S))})")


@torch.no_grad()
def part_measure(lab):
    from frontierlab.model import LM
    from frontierlab.model.config import toy
    from frontierlab.perf import benchmark
    kinds = {"gqa": toy(), "mla": toy().with_(attention="mla", extra={"kv_lora_rank": 64, "qk_rope_head_dim": 16}),
             "local_global (w=64)": toy().with_(attention="local_global",
                                                extra={"window": 64, "layer_types": ["local"] * 3 + ["global"]}),
             "hybrid GDN 3:1": toy().with_(attention="hybrid", extra={"linear_kind": "gdn", "full_every": 4})}
    print("Measured on this CPU (toy models, 4 layers, 128 wide, float32, batch 1; median of repeats)")
    print(f"  {'kind':22s} {'prefill/token':>14s} {'decode/token':>13s} {'ratio':>6s}  {'cache bytes at 512':>18s} "
          f"{'formula':>9s}  decode step at S=128 / 512 / 1024 (ms)")
    for name, cfg in kinds.items():
        torch.manual_seed(0)
        m = LM(cfg).eval()
        x = torch.randint(0, cfg.vocab_size, (1, 512))
        t_pre = benchmark(lambda: m(x, cache=m.new_cache()), warmup=2, repeats=5).median / 512
        steps = {}
        for S in (128, 512, 1024):
            h = {}

            def setup(S=S, h=h):
                c = m.new_cache()
                m(torch.randint(0, cfg.vocab_size, (1, S)), cache=c)
                h["c"] = c
            tok = torch.ones((1, 1), dtype=torch.long)
            steps[S] = benchmark(lambda h=h: m(tok, cache=h["c"]), warmup=3, repeats=30, setup=setup).median
        c = m.new_cache()
        m(x, cache=c)
        from frontierlab.attention import accounting as acc
        formula = (acc.m05_cache_bytes(cfg, 512, 4, 4, include_pos=True) if SV._is_m05(cfg)
                   else acc.cache_bytes(cfg, 512, 4))
        print(f"  {name:22s} {t_pre * 1e6:11.1f} us {steps[512] * 1e6:10.1f} us {steps[512] / t_pre:6.1f}  "
              f"{c.nbytes():18,d} {formula:9,.0f}  " + " / ".join(f"{steps[S] * 1e3:.2f}" for S in (128, 512, 1024)))
    print("  On a CPU both phases are bound by overheads and FLOPs, not by HBM; the ratio shows the batching win of "
          "prefill, not the GPU's memory-bound decode.")


ARMS = (("prefill-first x4", "prefill_first", None), ("chunked x4", "chunked", None),
        ("disagg 1P+3D", "disaggregated", 1), ("disagg 2P+2D", "disaggregated", 2))


def part_serve(lab):
    cfg = SV.QWEN3_1_7B
    ttft_slo, tpot_slo, eff = 0.5, 0.010, 0.5
    print(f"Iteration-level simulator, Stage D shape on 4x H100 (roofline x {eff} efficiency, PROJECTED); "
          f"SLOs: TTFT <= {ttft_slo * 1e3:.0f} ms, TPOT <= {tpot_slo * 1e3:.0f} ms; 400 Gb/s KV link")
    for wl, (P, O), rates in (("long prompt (RAG-like)", (8192, 256), (8, 20, 32)),
                              ("long answer (reasoning-like)", (512, 4096), (1, 3, 6))):
        print(f"  workload: {wl}, prompt {P}, output {O} (lengths +-30%)")
        print(f"    {'rate/s':>6s} {'arm':>17s} {'TTFT p50':>9s} {'TTFT p90':>9s} {'TPOT p50':>9s} {'TPOT p90':>9s} "
              f"{'goodput':>8s} {'tok/s':>8s}")
        for rate in rates:
            reqs = SV.poisson_requests(rate, 120 if O <= 256 else 60, P, O, seed=0, jitter=0.3)
            for label, pol, npre in ARMS:
                s = SV.simulate(reqs, cfg, policy=pol, gpus=4, chunk=2048, efficiency=eff, prefill_gpus=npre)
                g = SV.goodput(s, ttft_slo, tpot_slo)
                print(f"    {rate:6.1f} {label:>17s} {s['ttft_p50'] * 1e3:7.0f}ms {s['ttft_p90'] * 1e3:7.0f}ms "
                      f"{s['tpot_p50'] * 1e3:7.2f}ms {s['tpot_p90'] * 1e3:7.2f}ms {g:8.2f} {s['tokens_per_s']:8,.0f}")


def part_rollout(lab):
    hw = "H100-SXM"
    trainer_gib = SV.n_params(SV.QWEN3_1_7B) * 16 / GiB      # fp32 master weights + grads + AdamW moments
    print(f"One RL step's rollouts: 32 prompts x 8 samples = 256 sequences, prompt 512 tokens, on one H100 80 GB "
          f"colocated with the trainer ({trainer_gib:.1f} GiB of fp32 weights, gradients and AdamW state); "
          "roofline lower bound, equal-length responses (PROJECTED)")
    print(f"  {'design':30s} " + "".join(f"{f'gen {g}':>30s}" for g in (4096, 16384, 32768)))
    for name, cfg in SV.stage_d_designs().items():
        cells = []
        for gen in (4096, 16384, 32768):
            r = SV.rollout_time(cfg, 256, 512, gen, hw, trainer_gib=trainer_gib)
            cells.append(f"batch {r['batch']:3d} x{r['waves']:3d} {r['seconds']:7.0f} s" if r["batch"] else "does not fit")
        print(f"  {name:30s} " + "".join(f"{c:>30s}" for c in cells))


PARTS = {"kv": part_kv, "roofline": part_roofline, "measure": part_measure, "serve": part_serve, "rollout": part_rollout}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["all", *PARTS], default="all")
    ap.add_argument("--variant", choices=["cpu", "main", "t4"], default="cpu")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    if a.variant != "cpu":
        print("# not run in this build; part of the Module 15 pilot. vLLM 0.30.0 on the GPU, Qwen/Qwen3-1.7B-Base ea980cb:")
        print("vllm bench throughput --model Qwen/Qwen3-1.7B-Base --revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 "
              "--input-len 8192 --output-len 256 --num-prompts 200")
        print("vllm bench throughput --model Qwen/Qwen3-1.7B-Base --revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 "
              "--input-len 512 --output-len 4096 --num-prompts 200")
        print("# then compare measured tokens/s with --part roofline / serve; check vLLM's printed KV-cache capacity "
              "(GPU KV cache size: N tokens) against --part kv")
        return
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    t0 = time.perf_counter()
    for name, fn in PARTS.items():
        if a.part in ("all", name):
            print(f"\n== {name}")
            fn(lab)
    print(f"\ntotal {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
