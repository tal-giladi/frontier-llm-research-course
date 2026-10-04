"""Reference solution for lab 09.4."""

from __future__ import annotations

import math
from pathlib import Path


def young(delta, M):
    return math.sqrt(2.0 * delta * M)


def goodput(tau, delta, M, R):
    return tau / (M * math.exp(R / M) * (math.exp((tau + delta) / M) - 1.0))


def latest_committed(ckpt_root):
    root = Path(ckpt_root)
    if not root.exists():
        return None
    done = sorted(p for p in root.glob("step_*") if (p / "COMMITTED").exists())
    return done[-1] if done else None


def lost_work(records):
    out, last, seen_start = [], None, False
    for r in records:
        if r.get("event") == "start":
            if seen_start:
                out.append(max(0, (last if last is not None else r["from_step"]) - r["from_step"]))
            seen_start, last = True, None
        elif r.get("event") == "step":
            last = r["step"]
    return out
