"""Stability metrics for an RL run, computed from its ``metrics.jsonl`` (lessons 14.2, project).

"More stable" has to mean something measurable before runs start. The course uses, per run:

* ``entropy_drop`` — entropy over the first 10% of steps minus entropy over the last 10% (collapse shows as a
  large drop); ``entropy_min``;
* ``grad_norm_p99`` and ``grad_spikes`` — the 99th percentile of the pre-clip gradient norm, and the number of
  steps whose norm exceeds 5x the running median of the previous 20 steps;
* ``ratio_max`` — the largest importance ratio any update saw; ``clip_frac`` its mean;
* ``kl_final`` — exact KL to the reference over the last 10% of steps (on the toy);
* ``eval_drawdown`` — the largest fall of held-out accuracy from its running maximum (a run that learned and
  then lost it); ``collapsed`` = drawdown >= ``collapse_at``.
"""

from __future__ import annotations

import numpy as np

from frontierlab.metrics.jsonl import read_jsonl


def _vals(rows, key):
    return np.array([r[key] for r in rows if r.get(key) is not None], dtype=float)


def stability(rows: list[dict], collapse_at: float = 0.10, eval_key: str = "sampled_acc") -> dict:
    tr = [r for r in rows if r.get("split") == "train"]
    ev = [r for r in rows if r.get("split") == "eval"]
    n = len(tr)
    w = max(1, n // 10)
    ent = _vals(tr, "entropy")
    g = _vals(tr, "grad_norm")
    spikes = 0
    for i in range(20, g.size):
        if g[i] > 5 * np.median(g[i - 20:i]):
            spikes += 1
    acc = _vals(ev, eval_key)
    draw = float(np.max(np.maximum.accumulate(acc) - acc)) if acc.size else 0.0
    kl = _vals(tr[-w:], "kl_exact")
    rmax = _vals(tr, "ratio_max")
    cf = _vals(tr, "clip_frac")
    return {"entropy_drop": float(ent[:w].mean() - ent[-w:].mean()) if ent.size else float("nan"),
            "entropy_min": float(ent.min()) if ent.size else float("nan"),
            "grad_norm_p99": float(np.quantile(g, 0.99)) if g.size else float("nan"),
            "grad_spikes": int(spikes),
            "ratio_max": float(rmax.max()) if rmax.size else float("nan"),
            "clip_frac": float(cf.mean()) if cf.size else float("nan"),
            "kl_final": float(kl.mean()) if kl.size else float("nan"),
            "eval_drawdown": draw, "collapsed": bool(draw >= collapse_at)}


def run_stability(run_dir, **kw) -> dict:
    from pathlib import Path
    return stability(read_jsonl(Path(run_dir) / "metrics.jsonl"), **kw)
