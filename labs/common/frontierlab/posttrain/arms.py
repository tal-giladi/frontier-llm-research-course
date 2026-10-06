"""Run several RL arms over seeds and compare them at the seed level (labs 12.2, 12.3).

Every arm is a set of :class:`~frontierlab.posttrain.rl.RLConfig` overrides on one base configuration, so
the arms differ only in what the overrides name (the experiment contract's changed variable). Finished
runs (``policy.pt`` present) are skipped; interrupted runs resume exactly.

Comparisons are paired by seed (same seed = same initial policy, same first prompts and sampler stream),
with a t-interval over the per-seed differences: with 2-3 seeds that interval is wide, which is the point.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from frontierlab.metrics.jsonl import read_jsonl
from frontierlab.posttrain.rl import RLConfig, train


def run_arms(base: RLConfig, arms: dict[str, dict], seeds: list[int], root: str | Path) -> dict:
    root = Path(root)
    out = {}
    for name, over in arms.items():
        out[name] = []
        for s in seeds:
            cfg = replace(base, **over, seed=s, run=str(root / f"{name}-s{s}"))
            if not (Path(cfg.run) / "policy.pt").exists():
                print(f"== arm {name} seed {s}")
                train(cfg)
            out[name].append(summarise_run(cfg.run))
    return out


def summarise_run(run: str | Path, last: int = 20) -> dict:
    rows = read_jsonl(Path(run) / "metrics.jsonl")
    tr = [r for r in rows if r["split"] == "train"]
    ev = [r for r in rows if r["split"] == "eval"]

    def mean_last(key):
        vals = [r[key] for r in tr[-last:] if r.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    return {"final_greedy": ev[-1]["greedy_acc"], "final_sampled": ev[-1]["sampled_acc"],
            "final_len": ev[-1]["sampled_len"], "final_trunc": ev[-1]["sampled_trunc"],
            "train_pass_auc": float(np.mean([r["pass"] for r in tr])),
            **{k: mean_last(k) for k in ("entropy", "kl_exact", "kl_k3", "len", "len_wrong", "len_correct", "trunc",
                                         "clip_frac", "zero_var_frac", "ratio_max", "sampler_gap", "pass")},
            "curve_eval": [(r["step"], r["sampled_acc"]) for r in ev]}


def t_interval(x, alpha: float = 0.05) -> tuple[float, float, float]:
    from scipy.stats import t
    x = np.asarray(x, dtype=np.float64)
    m = float(x.mean())
    if x.size < 2:
        return m, float("nan"), float("nan")
    h = float(t.ppf(1 - alpha / 2, x.size - 1) * x.std(ddof=1) / math.sqrt(x.size))
    return m, m - h, m + h


def table(results: dict, metric: str, baseline: str | None = None) -> str:
    """Per arm: per-seed values, mean, and (if ``baseline``) the paired difference with its 95% t-interval."""
    lines = [f"{metric}"]
    for name, runs in results.items():
        vals = [r[metric] for r in runs]
        line = f"  {name:22s} " + " ".join(f"{v:7.3f}" for v in vals) + f"   mean {np.mean(vals):7.3f}"
        if baseline and name != baseline:
            d = np.array(vals) - np.array([r[metric] for r in results[baseline]])
            m, lo, hi = t_interval(d)
            line += f"   vs {baseline}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        lines.append(line)
    return "\n".join(lines)


def paired_diff(results: dict, arm: str, baseline: str, metric: str) -> dict:
    d = np.array([r[metric] for r in results[arm]]) - np.array([r[metric] for r in results[baseline]])
    m, lo, hi = t_interval(d)
    return {"mean": m, "ci": (lo, hi), "per_seed": d.tolist()}
