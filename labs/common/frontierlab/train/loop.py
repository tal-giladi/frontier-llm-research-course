"""The course training loop: CPU, one GPU, or a Colab session that may disconnect.

    python -m frontierlab.train.loop --run runs/b0-s0 --preset toy --steps 300               # CPU
    python -m frontierlab.train.loop --run runs/b0-s0 --preset baseline0 --steps 9500 \\
        --batch 32 --grad-accum 8 --seq 1024 --dtype bf16 --compile                         # main path

Exact resume: every checkpoint holds model, optimizer, step, the data-sampler RNG, the torch CPU RNG
and (on GPU) the CUDA RNG. Restarting with the same arguments continues the *same* run — the same
batches in the same order and the same learning rate at every step (``tests/test_train.py`` checks
this). ``--max-minutes`` stops cleanly and checkpoints before a session limit; rerun the same
command to continue.

Logs one JSON object per line to ``<run>/metrics.jsonl``; writes ``<run>/run_card.yaml`` (lesson
01.5). Later modules extend this loop (optimizers, precision, schedules) instead of replacing it.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from frontierlab.data.loader import TokenData
from frontierlab.evals.heldout import window_losses
from frontierlab.flops import PEAK_BF16, flops_per_token
from frontierlab.metrics.jsonl import JsonlLogger
from frontierlab.model import LM, PRESETS, param_counts
from frontierlab.runcard import write_run_card


def lr_at(step: int, steps: int, lr: float, warmup: int, schedule: str = "cosine", decay_frac: float = 0.2,
          min_ratio: float = 0.1) -> float:
    """Learning rate at ``step``: linear warmup, then cosine to ``min_ratio·lr``, WSD or constant.

    WSD (warmup-stable-decay, lesson 07.4): constant after warmup, then linear decay to
    ``min_ratio·lr`` over the last ``decay_frac`` of the run.
    """
    if step < warmup:
        return lr * (step + 1) / warmup
    if schedule == "constant":
        return lr
    if schedule == "wsd":
        decay_start = int(steps * (1 - decay_frac))
        if step < decay_start:
            return lr
        frac = (step - decay_start) / max(1, steps - decay_start)
        return lr * (1 - (1 - min_ratio) * frac)
    progress = (step - warmup) / max(1, steps - warmup)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="toy")
    ap.add_argument("--attention", default=None, help="override cfg.attention (Stage B branches)")
    ap.add_argument("--steps", type=int, default=300, help="length of the run (sets the LR schedule)")
    ap.add_argument("--stop-after", type=int, default=None, help="checkpoint and stop at this step")
    ap.add_argument("--max-minutes", type=float, default=None, help="checkpoint and stop after this wall time")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--schedule", choices=["cosine", "wsd", "constant"], default="cosine")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0, help="model init and data order")
    ap.add_argument("--data-seed", type=int, default=None, help="data order only (default: --seed)")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32", help="autocast dtype on GPU")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-windows", type=int, default=64)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--peak", default=None, help=f"GPU key for MFU logging: {sorted(PEAK_BF16)}")
    ap.add_argument("--question", default="", help="the question this run answers (goes in the run card)")
    ap.add_argument("--parent", default=None, help="run this one branches from (goes in the run card)")
    return ap


def make_model(a, vocab_size: int):
    cfg = PRESETS[a.preset](vocab_size=vocab_size)
    if a.attention:
        cfg = cfg.with_(attention=a.attention)
    return cfg, LM(cfg)


def make_optimizer(model, a):
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "norm" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 or "norm" in n]
    return torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95),
                             fused=a.device.startswith("cuda"))


def main(argv=None):
    a = build_parser().parse_args(argv)
    torch.manual_seed(a.seed)
    kw = {"root": a.data} if a.data else {}
    train, val = TokenData("train", **kw), TokenData("val", **kw)
    cfg, model = make_model(a, train.meta["vocab_size"])
    model.to(a.device)
    opt = make_optimizer(model, a)
    gen = torch.Generator().manual_seed(a.seed if a.data_seed is None else a.data_seed)
    a.run.mkdir(parents=True, exist_ok=True)
    ckpt_path, step = a.run / "checkpoint.pt", 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=a.device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        gen.set_state(ck["data_rng"])
        torch.set_rng_state(ck["torch_rng"])
        if ck.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        step = ck["step"]
        print(f"resumed from step {step}")
    tokens_per_step = a.batch * a.seq * a.grad_accum
    fpt = flops_per_token(cfg, a.seq)
    pc = param_counts(cfg)
    write_run_card(a.run, question=a.question, parent=a.parent, config=asdict(cfg), args=vars(a),
                   data_meta=train.meta, budget={"steps": a.steps, "tokens": a.steps * tokens_per_step,
                                                 "train_flops": fpt * a.steps * tokens_per_step},
                   extra={"params": pc})
    fwd = torch.compile(model) if a.compile else model
    autocast = torch.bfloat16 if (a.dtype == "bf16" and a.device.startswith("cuda")) else None
    log = JsonlLogger(a.run / "metrics.jsonl")

    def save():
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                    "data_rng": gen.get_state(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "config": asdict(cfg)}, ckpt_path)

    t_start = t_last = time.perf_counter()
    tokens_since = 0
    stop = min(a.steps, a.stop_after or a.steps)
    while step < stop:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, a.warmup, a.schedule)
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(a.grad_accum):
            x = train.batch(a.batch, a.seq, gen, a.device)
            with torch.autocast(device_type=x.device.type, dtype=autocast, enabled=autocast is not None):
                out = fwd(x, labels=x)
            (out.loss / a.grad_accum).backward()
            loss_sum += out.loss.item() / a.grad_accum
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        step += 1
        tokens_since += tokens_per_step
        if step % a.log_every == 0 or step == a.steps:
            if a.device.startswith("cuda"):
                torch.cuda.synchronize()
            dt = time.perf_counter() - t_last
            row = dict(split="train", step=step, loss=loss_sum, grad_norm=float(gnorm),
                       lr=opt.param_groups[0]["lr"], tok_per_s=round(tokens_since / dt),
                       tokens=step * tokens_per_step)
            if a.peak:
                row["mfu"] = round(fpt * tokens_since / dt / PEAK_BF16[a.peak], 4)
            if a.device.startswith("cuda"):
                row["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            log.log(**row)
            print(f"step {step:6d}  loss {loss_sum:.4f}  gnorm {float(gnorm):.2f}  {tokens_since / dt:,.0f} tok/s"
                  + (f"  mfu {row['mfu']:.3f}" if "mfu" in row else ""))
            t_last, tokens_since = time.perf_counter(), 0
        if step % a.eval_every == 0 or step == a.steps:
            losses = window_losses(model, val, a.eval_windows, a.seq, device=a.device, autocast_dtype=autocast)
            v = sum(losses) / len(losses)
            log.log(split="val", step=step, loss=v, windows=len(losses))
            print(f"step {step:6d}  val loss {v:.4f}")
            t_last = time.perf_counter()
        out_of_time = a.max_minutes is not None and (time.perf_counter() - t_start) / 60 > a.max_minutes
        if step % a.ckpt_every == 0 or step == stop or out_of_time:
            save()
        if out_of_time:
            print(f"stopped at step {step} after {a.max_minutes} minutes; rerun the same command to resume")
            break
    log.close()
    return model


if __name__ == "__main__":
    main()
