"""Decode memory and latency of several Module 3 arms at fixed contexts (lessons 03.1, 03.2; project).

    python labs/module-03/decode_compare.py --arms b0 gqa-kv mla mla-absorbed                     # free CPU
    python labs/module-03/decode_compare.py --arms b0 local-global --contexts 1024 4096 16384
    python labs/module-03/decode_compare.py --device cuda --dtype bf16 --preset baseline0 \
        --vocab 32768 --arms b0 gqa-kv mla-absorbed local-global --contexts 8192 16384 32768     # main path
    python labs/module-03/decode_compare.py --runs runs/m03/cpu/A-b0-s0 runs/m03/cpu/A-mla-s0       # trained runs

Arms are names from ``m03.ARMS``; ``mla-absorbed`` / ``mla-naive`` choose MLA's decode path. With random
weights (the default) only memory and time are meaningful; ``--runs`` loads trained checkpoints
instead (same numbers, plus a sanity check that the trained model decodes).

For each context S the script prints, per arm: bytes held by the cache (measured, ``Cache.nbytes``)
next to the formula (``frontierlab.attention.accounting``), and the median time of one decode step at
context S with a 95% interval, from interleaved rounds (lesson 02.4), with the paired speed-up versus
the first arm. On CUDA it adds the allocator's view of the cache and of one step's peak.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m03  # noqa: E402
from frontierlab.attention import accounting  # noqa: E402
from frontierlab.attention.bench import decode_benchmark  # noqa: E402
from frontierlab.attention.mla import set_mla_mode  # noqa: E402
from frontierlab.model import LM, PRESETS  # noqa: E402


def build(arm: str, preset: str, vocab: int, T: int, match: bool):
    base = PRESETS[preset](vocab_size=vocab)
    name, mode = (arm, None)
    if arm.startswith("mla-"):
        name, mode = "mla", arm.split("-", 1)[1]
    cfg = m03.arm_config(name, base, T, match_params=match)
    torch.manual_seed(0)
    model = LM(cfg)
    if mode:
        set_mla_mode(model, mode)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="*", default=["b0", "gqa-kv", "mla-naive", "mla-absorbed"])
    ap.add_argument("--runs", nargs="*", default=None, help="trained run folders instead of --arms")
    ap.add_argument("--preset", default="toy", choices=sorted(PRESETS))
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--train-seq", type=int, default=128, help="training T that sets the local-global window (T/4)")
    ap.add_argument("--match-params", action="store_true")
    ap.add_argument("--contexts", type=int, nargs="*", default=[512, 2048, 8192])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=1024, help="prefill chunk")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: torch's choice)")
    ap.add_argument("--out", default=None, help="write all rows as JSON")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    models = {}
    if a.runs:
        for r in a.runs:
            m, _ = m03.load_run(r, a.device)
            models[Path(r).name] = m
    else:
        for arm in a.arms:
            models[arm] = build(arm, a.preset, a.vocab, a.train_seq, a.match_params)
    for name, m in models.items():
        m.to(device=a.device, dtype=dt).eval()
        print(f"{name:<14s} {m03.describe(m.config, a.train_seq)}")
    print(f"\ndevice {a.device}, {a.dtype}, batch {a.batch}, torch {torch.__version__}, threads {torch.get_num_threads()}")
    rows = decode_benchmark(models, a.contexts, B=a.batch, warmup=a.warmup, rounds=a.rounds, chunk=a.chunk,
                            device=a.device)
    bpe = 2 if a.dtype == "bf16" else 4
    print("\nmeasured cache bytes vs formula (accounting.cache_bytes, includes the int64 positions):")
    for r in rows:
        f = accounting.cache_bytes(models[r["arm"]].config, r["S"], bytes_per=bpe, batch=a.batch)
        print(f"  S={r['S']:>7d} {r['arm']:<14s} measured {r['cache_bytes']:>12,d}  formula {f:>14,.0f}  "
              f"{'OK' if r['cache_bytes'] == f else 'MISMATCH'}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=2, default=str))
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
