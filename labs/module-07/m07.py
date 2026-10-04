"""Shared helpers for the Module 7 labs and project: run an arm, evaluate it, compare arms.

Lab scripts add ``labs/module-07`` to ``sys.path`` and ``import m07``. Every run goes through
``python -m frontierlab.optim.train`` (the course loop plus the Module 7 wrapper), so exact resume holds:
rerunning a script skips finished runs and continues interrupted ones.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from frontierlab.data.loader import TokenData
from frontierlab.evals.heldout import window_losses
from frontierlab.metrics import read_jsonl
from frontierlab.optim import train as optim_train
from frontierlab.stats import paired_bootstrap

CPU_EVAL = {"n": 256, "T": 128}


def finished(run: Path, steps: int) -> bool:
    log = run / "metrics.jsonl"
    if not log.exists():
        return False
    rows = [r for r in read_jsonl(log) if r["split"] == "train"]
    return bool(rows) and rows[-1]["step"] >= steps


def run_arm(run: Path, argv: list[str], steps: int) -> float:
    """Train one arm unless it already finished; returns the wall-clock seconds of this call."""
    if finished(run, steps):
        print(f"{run}: finished, skipping")
        return 0.0
    t0 = time.perf_counter()
    optim_train.main(["--run", str(run), "--steps", str(steps), *argv])
    dt = time.perf_counter() - t0
    with open(run / "wallclock.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"seconds": round(dt, 2), "time": round(time.time(), 1)}) + "\n")
    return dt


def wallclock(run: Path) -> float:
    """Total measured training seconds of a run (sum over its sessions)."""
    p = run / "wallclock.jsonl"
    return sum(r["seconds"] for r in read_jsonl(p)) if p.exists() else float("nan")


def eval_losses(run: Path, n: int = CPU_EVAL["n"], T: int = CPU_EVAL["T"], device: str = "cpu",
                data: str | None = None) -> list[float]:
    """Per-window held-out losses of the run's final checkpoint (cached in <run>/eval_<n>x<T>.json).

    ``run`` may also be a checkpoint file (cached next to it as <file>.eval_<n>x<T>.json)."""
    run = Path(run)
    if run.is_file():
        ckpt, cache = run, run.with_name(f"{run.stem}.eval_{n}x{T}.json")
    else:
        ckpt, cache = run / "checkpoint.pt", run / f"eval_{n}x{T}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    model = optim_train.load_model(ckpt, map_location=device).to(device)
    val = TokenData("val", **({"root": data} if data else {}))
    losses = window_losses(model, val, n, T, device=device)
    cache.write_text(json.dumps(losses))
    return losses


def stability(run: Path) -> list[dict]:
    p = run / "stability.jsonl"
    return read_jsonl(p) if p.exists() else []


def train_rows(run: Path) -> list[dict]:
    return [r for r in read_jsonl(run / "metrics.jsonl") if r["split"] == "train"]


def compare(a: list[float], b: list[float]) -> str:
    """'mean diff [lo, hi]' of a - b, paired by window (lesson 01.4)."""
    r = paired_bootstrap(a, b)
    return f"{r['mean_diff']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"


def mean(x) -> float:
    return float(np.mean(x))
