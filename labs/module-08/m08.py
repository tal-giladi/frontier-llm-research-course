"""Shared helpers for the Module 8 labs and project: run an arm, evaluate it, compare arms.

Lab scripts add ``labs/module-08`` to ``sys.path`` and ``import m08``. Every run goes through
``python -m frontierlab.precision.train`` (the course loop + the Module 7 wrapper + the Module 8 precision
swap), so exact resume holds: rerunning a script skips finished runs and continues interrupted ones.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from frontierlab.data.loader import TokenData
from frontierlab.evals.heldout import window_losses
from frontierlab.metrics import read_jsonl
from frontierlab.precision import train as prec_train
from frontierlab.stats import paired_bootstrap, summary

CPU_EVAL = {"n": 256, "T": 128}


def finished(run: Path, steps: int) -> bool:
    log = run / "metrics.jsonl"
    if not log.exists():
        return False
    rows = [r for r in read_jsonl(log) if r["split"] == "train"]
    return bool(rows) and rows[-1]["step"] >= steps


def run_arm(run: Path, argv: list[str], steps: int) -> float:
    """Train one arm unless it already finished; returns the wall-clock seconds of this call."""
    run = Path(run)
    if finished(run, steps):
        print(f"{run}: finished, skipping")
        return 0.0
    t0 = time.perf_counter()
    prec_train.main(["--run", str(run), "--steps", str(steps), *argv])
    dt = time.perf_counter() - t0
    with open(run / "wallclock.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"seconds": round(dt, 2), "time": round(time.time(), 1)}) + "\n")
    return dt


def wallclock(run: Path) -> float:
    p = Path(run) / "wallclock.jsonl"
    return sum(r["seconds"] for r in read_jsonl(p)) if p.exists() else float("nan")


def eval_losses(run: Path, n: int = CPU_EVAL["n"], T: int = CPU_EVAL["T"], device: str = "cpu", data: str | None = None,
                recipe: str | None = None, tag: str = "") -> list[float]:
    """Per-window held-out losses of a run's final checkpoint, evaluated with the precision it trained with
    (or ``recipe`` if given). Cached in ``<run>/eval_<n>x<T><tag>.json``."""
    run = Path(run)
    cache = run / f"eval_{n}x{T}{tag}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    model = prec_train.load_model(run / "checkpoint.pt", recipe=recipe, map_location=device).to(device)
    val = TokenData("val", **({"root": data} if data else {}))
    losses = window_losses(model, val, n, T, device=device)
    cache.write_text(json.dumps(losses))
    return losses


def eval_model(model, n: int = CPU_EVAL["n"], T: int = CPU_EVAL["T"], device: str = "cpu", data: str | None = None) -> list[float]:
    val = TokenData("val", **({"root": data} if data else {}))
    return window_losses(model, val, n, T, device=device)


def precision_rows(run: Path) -> list[dict]:
    p = Path(run) / "precision.jsonl"
    return read_jsonl(p) if p.exists() else []


def compare(a: list[float], b: list[float]) -> dict:
    """a - b paired by window (lesson 01.4): mean difference and 95% bootstrap interval."""
    r = paired_bootstrap(a, b)
    return {"diff": r["mean_diff"], "lo": r["ci"][0], "hi": r["ci"][1]}


def fmt(c: dict) -> str:
    return f"{c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"


def seed_summary(values) -> dict:
    return summary(values)


def mean(x) -> float:
    return float(np.mean(x))


def train_rows(run: Path) -> list[dict]:
    p = Path(run) / "metrics.jsonl"
    return [r for r in read_jsonl(p) if r["split"] == "train"] if p.exists() else []
