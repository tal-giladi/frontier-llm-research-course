"""METR-style time horizons and their uncertainty (lesson 18.3).

Kwa et al. (METR, 2025, arXiv 2503.14499) fit, per model, a logistic curve of task success against the log of the
time a human expert takes on the task:

    p(success | t) = sigmoid(beta * (log2 h - log2 t))

``h`` is the **50% time horizon**: the human task length at which the fitted success probability is 0.5. ``beta``
is the slope per doubling of task length. The 80% horizon comes from the same two parameters, h80 = h * 2^(-ln 4 / beta)
(sigmoid(x) = 0.8 at x = ln 4), so it is not an independent measurement (METR, *Clarifying limitations of time
horizon*, 2026-01-22). Uncertainty: a hierarchical bootstrap that resamples task families, then tasks within each
family, then runs within each task (the paper's three levels), refitting each time.

Data: METR's released runs (``github.com/METR/eval-analysis-public``, pinned in :data:`METR_RUNS`; the repository
states no licence, so the file is downloaded by the learner and never committed). Each row is one run with
``alias`` (model), ``task_family``, ``task_id``, ``human_minutes``, ``score_binarized`` and the task weights METR
publishes (``invsqrt_task_weight``, ``equal_task_weight``).

This is a re-implementation from the paper's description, not METR's code: weights, filters and the fitting
details of their pipeline may differ, and its numbers are compared with METR's published ones in the lab.
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from pathlib import Path

import numpy as np

METR_RUNS = {
    "repo": "METR/eval-analysis-public",
    "commit": "52cb829c7a2efb2d659285c4b1768d191d97f8d2",
    "path": "reports/time-horizon-1-1/data/raw/runs.jsonl",
    "bytes": 15_008_064,
    "sha256": "609f904f4b6ae32129388da89d036e00bac511ad94223d2be1e58fc2b45b55cd",
    "rows": 24_008,
    "licence": "none stated in the repository (checked 2026-10-07): download for analysis, do not redistribute",
    "checked": "2026-10-07",
}
METR_URL = f"https://raw.githubusercontent.com/{METR_RUNS['repo']}/{METR_RUNS['commit']}/{METR_RUNS['path']}"
METR_DATES = {"path": "data/external/release_dates.yaml", "bytes": 2442,
              "sha256": "317b92915df5bf935908567a857116bcf6f0c7686ef8b0840897ed04a29232bc"}
"""Model release dates METR uses for its trend fits, same repository and commit."""
METR_PUBLISHED = {"Claude 3.7 Sonnet (Inspect)": 60.4, "o3 (Inspect)": 119.7, "GPT-5 (Inspect)": 203.0,
                  "Claude Opus 4.5 (Inspect)": 293.0, "GPT-5.2": 352.0, "Claude Opus 4.6 (Inspect)": 719.0}
"""METR's published 50% horizons in minutes (Time Horizon 1.1, https://metr.org/assets/benchmark_results_1_1.yaml,
checked 2026-10-07), for comparison with this module's re-implementation."""


def _fetch(url: str, sha256: str, dest: str | Path, timeout: float = 180.0) -> Path:
    dest = Path(dest)
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == sha256:
        return dest
    with urllib.request.urlopen(url, timeout=timeout) as r:             # noqa: S310 - pinned https URL
        data = r.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"{url} does not match the pinned SHA-256")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def download(dest: str | Path, timeout: float = 180.0) -> Path:
    return _fetch(METR_URL, METR_RUNS["sha256"], dest, timeout)


def download_dates(dest: str | Path) -> Path:
    url = f"https://raw.githubusercontent.com/{METR_RUNS['repo']}/{METR_RUNS['commit']}/{METR_DATES['path']}"
    return _fetch(url, METR_DATES["sha256"], dest)


def load_dates(path: str | Path) -> dict[str, str]:
    import yaml
    return {k: str(v) for k, v in yaml.safe_load(Path(path).read_text(encoding="utf-8"))["date"].items()}


def load_runs(path: str | Path, alias: str | None = None, drop_fatal: bool = False) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if alias is not None and r["alias"] != alias:
                continue
            if drop_fatal and r.get("fatal_error_from"):
                continue
            rows.append({k: r.get(k) for k in ("alias", "task_family", "task_id", "human_minutes", "score_binarized",
                                                "invsqrt_task_weight", "equal_task_weight", "fatal_error_from", "task_source")})
    return rows


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_logistic(log2_t: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, l2: float = 1e-4,
                 iters: int = 50) -> tuple[float, float]:
    """Weighted logistic regression y ~ sigmoid(a + b * log2_t) by Newton's method. Returns (a, b)."""
    x = np.asarray(log2_t, float)
    y = np.asarray(y, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    w = w / w.mean()
    X = np.stack([np.ones_like(x), x], 1)
    theta = np.zeros(2)
    for _ in range(iters):
        p = sigmoid(X @ theta)
        g = X.T @ (w * (p - y)) + l2 * theta
        H = (X * (w * p * (1 - p))[:, None]).T @ X + l2 * np.eye(2)
        step = np.linalg.solve(H, g)
        theta -= step
        if np.abs(step).max() < 1e-10:
            break
    return float(theta[0]), float(theta[1])


def horizon_from(a: float, b: float, p: float = 0.5) -> float:
    """Human minutes at which the fitted success probability is ``p``. With b < 0 (success falls with length):
    a + b log2 t = logit(p)."""
    if b >= 0:
        return float("inf")
    return float(2.0 ** ((math.log(p / (1 - p)) - a) / b))


def fit_horizon(rows: list[dict], weight: str | None = "invsqrt_task_weight") -> dict:
    """50% and 80% horizons (minutes) and the slope beta = -b (per doubling of task length) for one model."""
    t = np.array([r["human_minutes"] for r in rows], float)
    y = np.array([r["score_binarized"] for r in rows], float)
    w = None if weight is None else np.array([r[weight] for r in rows], float)
    a, b = fit_logistic(np.log2(t), y, w)
    return {"h50": horizon_from(a, b, 0.5), "h80": horizon_from(a, b, 0.8), "beta": -b, "a": a, "n_runs": len(rows),
            "n_tasks": len({r["task_id"] for r in rows})}


def hierarchical_bootstrap(rows: list[dict], n_boot: int = 200, seed: int = 0, weight: str | None = "invsqrt_task_weight",
                           alpha: float = 0.05) -> dict:
    """Resample families, then tasks within each sampled family, then runs within each sampled task; refit."""
    rng = np.random.default_rng(seed)
    fam: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        fam.setdefault(r["task_family"], {}).setdefault(r["task_id"], []).append(r)
    fams = list(fam)
    h50, h80 = [], []
    for _ in range(n_boot):
        sample = []
        for f in rng.choice(len(fams), len(fams)):
            tasks = fam[fams[f]]
            ids = list(tasks)
            for ti in rng.choice(len(ids), len(ids)):
                runs = tasks[ids[ti]]
                sample.extend(runs[j] for j in rng.choice(len(runs), len(runs)))
        res = fit_horizon(sample, weight)
        h50.append(res["h50"])
        h80.append(res["h80"])
    q = [alpha / 2, 1 - alpha / 2]
    return {"h50_ci": tuple(float(v) for v in np.quantile(h50, q)), "h80_ci": tuple(float(v) for v in np.quantile(h80, q)),
            "n_boot": n_boot}


def doubling_time_days(release_days: list[float], horizons: list[float]) -> float:
    """Days per doubling from a least-squares fit of log2(horizon) on release date (days)."""
    x = np.asarray(release_days, float)
    y = np.log2(np.asarray(horizons, float))
    slope = np.polyfit(x, y, 1)[0]
    return float(1.0 / slope)


__all__ = ["METR_RUNS", "METR_URL", "METR_DATES", "METR_PUBLISHED", "download", "download_dates", "load_dates", "load_runs", "sigmoid", "fit_logistic", "horizon_from", "fit_horizon",
           "hierarchical_bootstrap", "doubling_time_days"]
