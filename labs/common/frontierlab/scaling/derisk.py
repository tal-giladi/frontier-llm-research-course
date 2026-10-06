"""De-risking a run (lesson 11.3 and the Recipe-R project): pre-registration, a predicted loss curve, go/no-go.

The point of a ladder is to be wrong cheaply. Three tools make "wrong" checkable:

* :func:`preregister` writes the prediction for the big run *before* it starts: point, interval, the whole
  predicted validation curve, the decision rules, and a digest of the ladder results it came from. The file is
  hashed and never overwritten; :func:`check_after` refuses a prediction written after the run's first log line.
* :func:`curve_band` predicts the big run's validation loss at fixed fractions of its schedule. Each ladder run
  is read at the same fractions of *its own* schedule; at each fraction ``f`` the law
  ``L_f = E_f + A_f·N^-α + B_f·D^-β`` is refitted with the exponents of the final fit held fixed (only E, A, B,
  by linear least squares), and bootstrapped over ladder runs for an interval. ``D`` is the run's planned total
  tokens: all runs share the schedule shape, so the same fraction means the same point of the schedule.
* :func:`monitor` compares the running big run with that band and with its own gradient-norm history and says
  when a pre-stated rule would have stopped it.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from frontierlab.scaling import fit as sfit


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()


def ladder_digest(runs: list[dict]) -> str:
    """Digest of the ladder results a prediction used (run names, sizes, tokens, final losses)."""
    keys = ("run", "N_nonemb", "N_total", "D", "C", "loss")
    return _digest([{k: r.get(k) for k in keys} for r in sorted(runs, key=lambda r: r["run"])])


def preregister(path, record: dict) -> dict:
    """Write ``record`` with a UTC timestamp and its SHA-256 to ``path``. Never overwrites an existing file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists: a pre-registration is written once (write a new, dated file instead)")
    body = {**record, "created_unix": round(time.time(), 3),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = {"body": body, "sha256": _digest(body)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=float))
    return out


def load_prereg(path) -> dict:
    d = json.loads(Path(path).read_text())
    if _digest(d["body"]) != d["sha256"]:
        raise ValueError(f"{path}: content does not match its digest (edited after it was written)")
    return d["body"]


def first_log_time(run_dir) -> float:
    from frontierlab.metrics.jsonl import read_jsonl
    rows = read_jsonl(Path(run_dir) / "metrics.jsonl")
    return min(float(r["time"]) for r in rows)


def check_after(prereg_path, run: dict, run_dir=None) -> dict:
    """Compare a finished run with its pre-registration: inside / above / below the interval, error, and the
    timing check (the prediction must be older than the run's first metrics line)."""
    body = load_prereg(prereg_path)
    p = body["prediction"]
    measured = run["loss"]
    timing_ok = None
    if run_dir is not None:
        timing_ok = body["created_unix"] < first_log_time(run_dir)
    where = "inside" if p["lo"] <= measured <= p["hi"] else ("above" if measured > p["hi"] else "below")
    return {"predicted": p["loss"], "interval": (p["lo"], p["hi"]), "level": p.get("level"), "measured": measured,
            "error": measured - p["loss"], "where": where, "timing_ok": timing_ok}


def loss_at_fraction(curve, steps: int, f: float) -> float:
    """Validation loss of a run at fraction ``f`` of its steps (the evaluation nearest to round(f·steps))."""
    target = f * steps
    s, v = min(curve, key=lambda sv: abs(sv[0] - target))
    if abs(s - target) > 0.05 * steps + 1:
        return math.nan
    return v


def curve_band(runs: list[dict], N: float, D: float, fractions, alpha: float, beta: float, key_N: str = "N_total",
               n_boot: int = 300, level: float = 0.9, seed: int = 0) -> list[dict]:
    """Predicted (lo, mid, hi) validation loss of a run with (N, D) at each fraction of its schedule."""
    rng = np.random.default_rng(seed)
    out = []
    for f in fractions:
        rows = [(r[key_N], r["D"], loss_at_fraction(r["val_curve"], r["steps"], f)) for r in runs]
        rows = np.array([x for x in rows if math.isfinite(x[2])])
        if len(rows) < 4:
            out.append({"fraction": f, "lo": math.nan, "mid": math.nan, "hi": math.nan})
            continue
        fix = {"alpha": alpha, "beta": beta}
        point = float(sfit.predict(sfit.fit_parametric(rows[:, 0], rows[:, 1], rows[:, 2], fix=fix), N, D))
        preds = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(rows), len(rows))
            if len(set(idx.tolist())) < 3:
                continue
            fb = sfit.fit_parametric(rows[idx, 0], rows[idx, 1], rows[idx, 2], fix=fix)
            preds.append(float(sfit.predict(fb, N, D)))
        lo, hi = np.quantile(preds, [(1 - level) / 2, (1 + level) / 2])
        out.append({"fraction": float(f), "lo": float(min(lo, point)), "mid": point, "hi": float(max(hi, point))})
    return out


def monitor(val_curve, steps: int, band: list[dict], tol: float = 0.02, grad_norms=None, spike_factor: float = 4.0,
            min_fraction: float = 0.1) -> dict:
    """Apply two pre-stated stopping rules to a running (or finished) run.

    1. **Off the band:** at an evaluation at fraction ≥ ``min_fraction`` (less 0.005, so the evaluation at step
       ``steps // 10`` counts as the 10% point), the validation loss is above the band's
       upper edge by more than ``tol`` at that fraction (band values interpolated between fractions).
    2. **Gradient spike:** a logged gradient norm exceeds ``spike_factor`` × the median of the norms logged before it
       (after the first 10% of the run, when norms settle).

    Returns the first step at which each rule fired (or None), the decision ("stop" if either fired) and the
    fraction of the run's compute that stopping there would have saved.
    """
    fr = np.array([b["fraction"] for b in band])
    hi = np.array([b["hi"] for b in band])
    ok = np.isfinite(hi)
    off = None
    for s, v in sorted(val_curve):
        f = s / steps
        if f + 0.005 < min_fraction or not ok.any():          # 0.005: an evaluation at steps//10 is 'the 10% point'
            continue
        h = float(np.interp(f, fr[ok], hi[ok]))
        if v > h + tol:
            off = {"step": int(s), "fraction": f, "loss": v, "band_hi": h}
            break
    spike = None
    if grad_norms:
        g = sorted(grad_norms)
        for i, (s, n) in enumerate(g):
            if s / steps < 0.1 or i < 3:
                continue
            med = float(np.median([x for _, x in g[:i]]))
            if n > spike_factor * med:
                spike = {"step": int(s), "fraction": s / steps, "grad_norm": n, "median_before": med}
                break
    fired = [x for x in (off, spike) if x is not None]
    first = min(fired, key=lambda x: x["step"]) if fired else None
    return {"off_band": off, "grad_spike": spike, "decision": "stop" if first else "continue",
            "stop_step": first["step"] if first else None,
            "compute_saved": 1 - first["step"] / steps if first else 0.0}


def go_no_go(checks: dict) -> dict:
    """``checks``: name -> (passed: bool, detail). Go only if every check passed; list the failures."""
    failed = {k: v[1] for k, v in checks.items() if not v[0]}
    return {"decision": "go" if not failed else "no-go", "failed": failed,
            "passed": [k for k, v in checks.items() if v[0]]}
