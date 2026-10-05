"""Scoring Module 10 runs on several held-out sets at once, and comparing arms by seed (lessons 10.1–10.5).

    python -m frontierlab.datax.evaluate RUN [RUN ...] --sets edu=labs/common/data/v0 web=labs/common/data/m10/web \\
        --windows 128 --seq 128 --lambada 1000

For each run it loads ``<run>/checkpoint.pt`` and writes ``<run>/eval_m10.json`` with, per held-out
set, the per-window losses of Eval v0's fixed windows (``TokenData.eval_windows``, seed 1234) on that
set's **validation** split, and the per-passage LAMBADA target log-probabilities (Eval v0's pinned
file, the first ``--lambada`` passages). Every number is per item, so arms can be paired.

Data decisions are claims about a *recipe*, so the replicate is the seed (lesson 01.4):
:func:`seed_level` takes per-seed means of two arms with matching seeds and returns the paired-by-seed
t-interval of the difference; :func:`window_paired` gives the paired bootstrap over items for one
seed pair (it treats the two models as fixed and is reported only alongside).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from frontierlab.data.loader import TokenData
from frontierlab.data.prepare import DEFAULT_OUT
from frontierlab.model import LM, ModelConfig

T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         11: 2.201, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def t975(df: int) -> float:
    if df in T_975:
        return T_975[df]
    keys = sorted(T_975)
    return T_975[min(keys, key=lambda k: abs(k - df))] if df < 30 else 1.96


def load_run(run: str | Path, device="cpu"):
    import frontierlab.datax.packing  # noqa: F401  (registers "gqa-docmask")
    ck = torch.load(Path(run) / "checkpoint.pt", map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["config"])
    m = LM(cfg)
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), cfg, ck


def score_run(run, sets: dict, windows: int = 128, seq: int = 128, n_lambada: int = 0, device="cpu",
              doc_mask: bool | None = None) -> dict:
    from frontierlab.datax import packing
    from frontierlab.evals.heldout import window_losses
    model, cfg, ck = load_run(run, device)
    masked = cfg.attention == "gqa-docmask" if doc_mask is None else doc_mask
    out = {"run": str(run), "step": int(ck["step"]), "windows": windows, "seq": seq, "window_seed": 1234,
           "doc_mask": masked, "sets": {}}
    for name, root in sets.items():
        val = TokenData("val", root)
        n = min(windows, (len(val.tokens) - 1) // seq)
        fn = packing.window_losses_docmask if masked else window_losses
        out["sets"][name] = {"root": str(root), "losses": fn(model, val, n, seq, device=device)}
    if n_lambada:
        from frontierlab.evals import suite_v0
        path = suite_v0.download_lambada()
        enc = suite_v0.data_v0_encoder()
        texts = suite_v0.load_lambada(path, n_lambada)
        items = suite_v0.lambada_scores(model, enc, texts, max_len=max(seq, 64), device=device)
        out["lambada"] = {"n": len(items), "logprob": [it["logprob"] for it in items],
                          "correct": [it["correct"] for it in items]}
    return out


def mean_of(res: dict, metric: str) -> float:
    """``metric`` is a set name (mean held-out loss) or ``lambada`` (mean target log-probability)."""
    if metric == "lambada":
        return float(np.mean(res["lambada"]["logprob"]))
    if metric == "avg":
        return float(np.mean([np.mean(v["losses"]) for v in res["sets"].values()]))
    return float(np.mean(res["sets"][metric]["losses"]))


def seed_level(a: list[float], b: list[float]) -> dict:
    """Paired-by-seed difference b − a: mean, 95% t-interval, per-seed differences (needs >= 2 seeds)."""
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    n = d.size
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    half = t975(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean_diff": float(d.mean()), "ci": (float(d.mean() - half), float(d.mean() + half)),
            "per_seed": d.tolist(), "sd_diff": sd}


def window_paired(a_items, b_items, n_boot: int = 4000) -> dict:
    from frontierlab.stats import paired_bootstrap
    return paired_bootstrap(b_items, a_items, n_boot=n_boot)


def decide(ci: tuple[float, float], margin: float, lower_is_better: bool = True) -> str:
    """Adopt if the whole interval shows an improvement larger than ``margin``; reject if the whole interval
    shows no improvement of that size (or harm); else inconclusive. ``ci`` is of (new − baseline)."""
    lo, hi = ci
    if not lower_is_better:
        lo, hi = -hi, -lo
    if hi < -margin:
        return "adopt"
    if lo > -margin:
        return "reject"
    return "inconclusive"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--sets", nargs="+", default=[f"edu={DEFAULT_OUT}"], help="name=prepared_folder")
    ap.add_argument("--windows", type=int, default=128)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lambada", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-name", default="eval_m10.json")
    a = ap.parse_args(argv)
    sets = dict(s.split("=", 1) for s in a.sets)
    for r in a.runs:
        res = score_run(r, sets, a.windows, a.seq, a.lambada, a.device)
        (r / a.out_name).write_text(json.dumps(res))
        msg = "  ".join(f"{k} {np.mean(v['losses']):.4f}" for k, v in res["sets"].items())
        if "lambada" in res:
            msg += f"  lambada_logprob {np.mean(res['lambada']['logprob']):.3f}"
        print(f"{r}: {msg}")


if __name__ == "__main__":
    main()
