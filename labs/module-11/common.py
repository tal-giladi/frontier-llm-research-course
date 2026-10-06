"""Shared helpers for the Module 11 lab scripts: run a planned ladder (skipping finished runs), read it back.

Not a lab file: every lesson script imports it as ``labs/module-11/common.py``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
try:                                      # Windows consoles default to cp1252; the tables print a few symbols
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
RUNS = Path(os.environ.get("M11_RUNS", REPO / "runs" / "m11"))

from frontierlab.scaling import ladder  # noqa: E402
from frontierlab.scaling import train as strain  # noqa: E402


def run_args(p: dict, v: dict, run: Path, seed: int = 0, lr: float | None = None, extra: list[str] | None = None) -> list[str]:
    """Loop arguments of one ladder run: warmup 5% of the steps (Porian et al. scale warmup with the run),
    cosine to 0.1×, ten evaluations per run on the same fixed windows, a checkpoint every half run."""
    steps = p["steps"]
    args = ["--run", str(run), "--preset", p["preset"], "--steps", str(steps), "--batch", str(v["batch"]),
            "--seq", str(v["seq"]), "--lr", str(lr if lr is not None else v["lr"]), "--warmup", str(max(5, steps // 20)),
            "--seed", str(seed), "--eval-every", str(max(1, steps // 10)), "--eval-windows", str(v["windows"]),
            "--log-every", str(max(1, steps // 50)), "--ckpt-every", str(max(1, steps // 2)), "--device", v["device"]]
    if v.get("dtype"):
        args += ["--dtype", v["dtype"]]
    if v.get("grad_accum", 1) > 1:
        args += ["--grad-accum", str(v["grad_accum"])]
    return args + (extra or [])


def ensure_run(run: Path, args: list[str], via: list[str] | None = None, print_only: bool = False) -> dict | None:
    """Train ``run`` unless it already finished (an interrupted run resumes exactly); return its results."""
    cmd = (via or []) + args
    if print_only:
        print("python -m frontierlab.scaling.train " + " ".join(cmd))
        return None
    if run.exists() and (run / "metrics.jsonl").exists():
        try:
            r = ladder.read_run(run)
            if r["finished"]:
                return r
        except Exception:  # noqa: BLE001 - incomplete card: rerun (resumes from the checkpoint)
            pass
    t0 = time.time()
    _quiet(lambda: strain.main(cmd))
    r = ladder.read_run(run)
    print(f"  {run.name}: {r['steps']} steps, loss {r['loss']:.4f}, {time.time() - t0:.0f} s", flush=True)
    return r


def _quiet(fn):
    """Run fn with the loop's per-step prints suppressed (they go to metrics.jsonl anyway)."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn()


def fmt(x: float, d: int = 3) -> str:
    return f"{x:.{d}g}" if abs(x) >= 1e4 or (abs(x) < 1e-3 and x != 0) else f"{x:.{d}f}"


def lab_module(argv_lab: str | None, here: Path):
    """Load the learner's lab.py (or $LAB_TARGET.py) next to a script."""
    from frontierlab.labkit import load_path
    name = argv_lab or os.environ.get("LAB_TARGET", "lab")
    return load_path(str(here / f"{name}.py"))


__all__ = ["REPO", "RUNS", "run_args", "ensure_run", "fmt", "lab_module", "sys"]
