"""Lab 15.4 (extension): what KV-cache and weight quantisation cost in quality, and what they buy in memory.

    python labs/module-15/lesson-04/quant_lab.py                    # free CPU, about 5 minutes (after lab 15.2)
    python labs/module-15/lesson-04/quant_lab.py --variant main --print

Uses the 15.2 target model (trained if missing; ``frontierlab.ttc.spec_models``). Emulation: every format is
quantise-dequantise in float32, so the numbers are about quality, never about speed.

1. Your quantiser and KIVI layout against the course's.
2. Where the outliers are: per-channel and per-token spread of the cached keys and values on real text.
3. KV formats on 48 validation windows of 256 tokens, decoded one token at a time with the cache: mean NLL
   change against the full-precision cache (paired bootstrap over windows) and mean KL of the next-token
   distributions, for 8/4/3/2 bits, keys per channel vs per token, residual window 0 vs 32 tokens.
4. Weight round-to-nearest formats on 64 held-out windows (teacher-forced): NLL change and storage bits.
5. Memory: the Stage D model's cache at 128K and its weights in each format, and what that does to batch
   capacity on one 80 GB GPU (calculated).
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path

import numpy as np
import torch

from frontierlab.labkit import load_path
from frontierlab.stats import paired_bootstrap
from frontierlab.ttc import kvquant as KQ
from frontierlab.ttc import serving as SV
from frontierlab.ttc import spec_models as SM

HERE = Path(__file__).resolve().parent
GiB = 2 ** 30


def windows(n: int, T: int, seed: int = 99) -> torch.Tensor:
    from frontierlab.data.loader import TokenData
    val = TokenData("val")
    return torch.stack([val.window(s, T) for s in val.eval_windows(n, T, seed)])


@torch.no_grad()
def outliers(model, x: torch.Tensor):
    """Spread of cached keys/values of layer 0..L-1: max |.| per channel over tokens, and per token over channels."""
    c = model.new_cache()
    model(x, cache=c)
    rows = []
    for i, lc in enumerate(c.layers):
        k, v = lc["k"].float(), lc["v"].float()
        kc = k.abs().amax(dim=(0, 2)).flatten()            # per (head, channel)
        vc = v.abs().amax(dim=(0, 2)).flatten()
        kt = k.abs().amax(dim=(1, 3)).flatten()            # per token
        rows.append((i, float(kc.max() / kc.median()), float(vc.max() / vc.median()), float(kt.max() / kt.median())))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["cpu", "main", "t4"], default="cpu")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    if a.variant != "cpu":
        print("# not run in this build; part of the Module 15 pilot (vLLM 0.30.0, 1x H100):")
        print("vllm bench throughput --model Qwen/Qwen3-1.7B-Base --revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 "
              "--input-len 512 --output-len 8192 --num-prompts 256 --kv-cache-dtype auto")
        print("vllm bench throughput --model Qwen/Qwen3-1.7B-Base --revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 "
              "--input-len 512 --output-len 8192 --num-prompts 256 --kv-cache-dtype fp8")
        print("# quality: python -m frontierlab.pipeline.hf_eval score on the same model served with each KV dtype "
              "(GSM8K strict + LAMBADA); weights: an AWQ/GPTQ int4 checkpoint per the inference course, Module 4")
        return
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    t0 = time.perf_counter()
    g = torch.Generator().manual_seed(0)
    k = torch.randn(1, 2, 100, 32, generator=g)
    v = torch.randn(1, 2, 100, 32, generator=g)
    a1, b1 = lab.kivi_qdq(k, v, 2, 32, 32)
    a2, b2 = KQ.kivi_qdq(k, v, 2, 32, 32)
    assert torch.allclose(a1, a2) and torch.allclose(b1, b2), "your kivi_qdq disagrees with the course's"
    print("check: your asym_qdq / kivi_qdq agree with frontierlab.ttc.kvquant")

    base = KQ.plain_copy(SM.ensure_target())
    x = windows(48, 256)
    print("\nOutliers in the cache (layer: max/median of per-channel max |k|, of per-channel max |v|, of per-token max |k|)")
    for i, kc, vc, kt in outliers(base, x[:8]):
        print(f"  layer {i}: keys per channel {kc:5.2f}x   values per channel {vc:5.2f}x   keys per token {kt:5.2f}x")

    ref = KQ.decode_nll(base, x, prefill=1)
    print(f"\nKV formats (48 windows x 255 decoded tokens; full-precision NLL {np.mean(ref['nll']):.4f}); "
          "dNLL > 0 is worse")
    print(f"  {'format':38s} {'bits/elt':>8s} {'dNLL':>8s}  {'95% CI':18s} {'mean KL':>9s}")
    for bits in (8, 4, 3, 2):
        for axis in ("channel", "token"):
            for res in (32, 0):
                if axis == "token" and res == 0:
                    continue
                m = KQ.with_quant_kv(base, bits=bits, group=32, residual=res, key_axis=axis)
                r = KQ.decode_nll(m, x, prefill=1, ref=base)
                d = paired_bootstrap(r["nll"], ref["nll"], n_boot=4000)
                name = f"int{bits}, keys per {axis}, residual {res}"
                print(f"  {name:38s} {KQ.bits_per_element(bits, 32):8.2f} {d['mean_diff']:+8.4f}  "
                      f"[{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] {np.mean(r['kl']):9.2e}")

    from frontierlab.data.loader import TokenData
    from frontierlab.evals.heldout import window_losses
    val = TokenData("val")
    w_ref = window_losses(base, val, 64, 256)
    print(f"\nWeight round-to-nearest (64 windows x 256, teacher-forced; full precision {np.mean(w_ref):.4f})")
    print(f"  {'format':30s} {'bits/weight':>11s} {'rel. error':>10s} {'dNLL':>8s}  95% CI")
    for bits, group, sym in ((8, 128, False), (4, 128, False), (4, 32, False), (4, 128, True), (3, 32, False), (2, 32, False)):
        m = copy.deepcopy(base)
        info = KQ.quantize_weights_(m, bits, group, symmetric=sym)
        w = window_losses(m, val, 64, 256)
        d = paired_bootstrap(w, w_ref, n_boot=4000)
        name = f"int{bits} g{group} {'symmetric' if sym else 'asymmetric'}"
        print(f"  {name:30s} {info['bits_per_weight']:11.2f} {info['mean_rel_err']:10.4f} {d['mean_diff']:+8.4f}  "
              f"[{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]")

    cfg = SV.QWEN3_1_7B
    S = 131072
    e = 2 * cfg.num_key_value_heads * cfg.head_dim
    print(f"\nStage D shape at {S:,} tokens on one 80 GB GPU (calculated; residual 128, G = 32)")
    print(f"  {'KV format':24s} {'GiB/seq':>8s}   sequences that fit with BF16 / INT4 weights (yours: GiB/seq)")
    for label, bits in (("BF16", 16), ("FP8", 8), ("KIVI-4", 4), ("KIVI-2", 2)):
        if bits >= 8:
            kv = cfg.num_hidden_layers * S * e * bits / 8
        else:
            kv = cfg.num_hidden_layers * KQ.kv_cache_bits(S, e, bits, 32, 128) / 8
        mine = cfg.num_hidden_layers * lab.kv_cache_bytes(S, e, bits, 32, 128) if bits < 8 else kv
        fit16 = SV.capacity(80 * GiB, SV.weight_bytes(cfg, 2), kv)
        fit4 = SV.capacity(80 * GiB, SV.weight_bytes(cfg, 4.25 / 8), kv)
        print(f"  {label:24s} {kv / GiB:8.2f}   {fit16:4d} / {fit4:4d}   ({mine / GiB:.2f})")
    print(f"\ntotal {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
