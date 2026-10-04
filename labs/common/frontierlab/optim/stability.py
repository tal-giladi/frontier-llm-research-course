"""Stability statistics during training, and reading them after a failure (lesson 07.5).

:class:`StabilityLogger` writes one JSON row per logged optimizer step to ``<run>/stability.jsonl``:

    step                   optimizer step just taken (1-based, as the loop's ``step``)
    max_logit              largest attention logit q·k/sqrt(d) over layers and heads on this step's batch
                           (before QK-Clip, if QK-Clip is on); ``max_logit_layers`` per layer
    clipped_heads          heads QK-Clip rescaled on this step (only with QK-Clip)
    ratio_<kind>_median    median over matrices of RMS(update) / RMS(weight), kind = muon or adamw
    ratio_max, ratio_max_name   the largest ratio and which parameter had it
    grad_norm_post         total gradient norm after the loop's clipping (the loop logs the norm before it)
    logz_mean, logz_max    mean and max |log Z| of the output softmax (z-loss watches log Z)
    logit_max              largest |output logit|

The loop itself logs ``loss`` and ``grad_norm`` (before clipping) to ``metrics.jsonl``; run with
``--log-every 1`` when you want per-step losses for spike detection. The logger does not change training:
straight and stop-and-resume runs are bit-identical with it on (tests/test_optim.py).

Reading the logs: :func:`detect_spikes` finds loss spikes against a rolling median; :func:`precursor`
finds when a statistic first crossed a threshold before a step; :func:`diagnose` applies the decision rules
of lesson 07.5 to say which failure a spike most resembles.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from frontierlab.optim.qkclip import LogitMonitor


class StabilityLogger:
    """Attach with ``StabilityLogger(model, opt, path, every=1, qkclip=clip)`` after QK-Clip (if any) is attached."""

    def __init__(self, model, opt, path, every: int = 1, qkclip=None, monitor: LogitMonitor | None = None):
        self.model, self.opt, self.every, self.qkclip = model, opt, max(1, int(every)), qkclip
        self.monitor = qkclip.monitor if qkclip is not None else (monitor or LogitMonitor(model))
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")
        self._armed = False
        model.register_forward_pre_hook(self._arm)
        opt.post_step_hooks.append(self)

    def _arm(self, module, args):
        if not module.training:
            return
        on = (self.opt.steps_taken + 1) % self.every == 0
        self._armed = on
        self.opt.collect_stats = on
        if self.qkclip is None:
            self.monitor.active = on
        if hasattr(self.model, "track_logz"):
            self.model.track_logz = on

    def __call__(self, opt):
        if not self._armed:
            return
        row = {"split": "stability", "step": opt.steps_taken}
        if self.qkclip is not None:
            row.update(max_logit=self.qkclip.last.get("max_logit"), clipped_heads=self.qkclip.last.get("clipped_heads"))
            row["max_logit_layers"] = self.qkclip.last.get("layers")
        else:
            smax = self.monitor.take()
            if smax:
                per = [round(float(smax[i].max()), 4) for i in sorted(smax)]
                row.update(max_logit=max(per), max_logit_layers=per)
        stats = opt.last_stats
        by_kind: dict[str, list] = {}
        best = (0.0, None)
        for name, s in stats.items():
            r = s["update_rms"] / max(s["weight_rms"], 1e-12)
            by_kind.setdefault(s["kind"], []).append(r)
            if r > best[0]:
                best = (r, name)
        for kind, vals in by_kind.items():
            row[f"ratio_{kind}_median"] = statistics.median(vals)
        row["ratio_max"], row["ratio_max_name"] = best
        grads = [p.grad for g in opt.param_groups for p in g["params"] if p.grad is not None]
        if grads:
            row["grad_norm_post"] = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float())
                                                                          for g in grads])).item()
        lz = getattr(self.model, "last_logz", None)
        if lz:
            row.update(lz)
        self._f.write(json.dumps(row) + "\n")
        self._f.flush()
        self._armed = False
        opt.collect_stats = False
        if self.qkclip is None:
            self.monitor.active = False

    def close(self):
        self._f.close()


# ------------------------------------------------------------------------------------------ forensics

def read_rows(path) -> list[dict]:
    """JSONL rows; for duplicated steps (a crash after the last checkpoint, then a resume) the last row wins."""
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[(r.get("split"), r["step"])] = r
    return sorted(rows.values(), key=lambda r: (str(r.get("split")), r["step"]))


def series(rows: list[dict], key: str, split: str | None = None) -> tuple[list[int], list[float]]:
    pts = [(r["step"], r[key]) for r in rows if key in r and r[key] is not None and (split is None or r.get("split") == split)]
    return [s for s, _ in pts], [float(v) for _, v in pts]


def smooth(values, width: int = 5) -> list[float]:
    """Centred running median over ``width`` points (shorter at the ends). Single-batch losses are noisy; a
    median keeps a spike of >= width/2 steps and a level shift, and removes one-step noise."""
    h = width // 2
    return [statistics.median(values[max(0, i - h):i + h + 1]) for i in range(len(values))]


def detect_spikes(steps, values, window: int = 20, k: float = 6.0, min_rel: float = 0.05) -> list[dict]:
    """Loss spikes: points above the rolling median of the previous ``window`` points by more than
    max(k · 1.4826 · MAD, min_rel · median). MAD = median absolute deviation (1.4826·MAD ~ std for normal noise).

    Consecutive flagged points are merged into one spike (start, peak, end).
    """
    flagged = []
    for i in range(window, len(values)):
        prev = values[i - window:i]
        med = statistics.median(prev)
        mad = statistics.median(abs(v - med) for v in prev)
        thr = max(k * 1.4826 * mad, min_rel * abs(med))
        if values[i] - med > thr:
            flagged.append((i, med, thr))
    spikes, cur = [], None
    for i, med, thr in flagged:
        if cur is not None and i == cur["last_i"] + 1:
            cur["last_i"] = i
            if values[i] > cur["peak_loss"]:
                cur.update(peak_step=steps[i], peak_loss=values[i])
        else:
            if cur is not None:
                spikes.append(cur)
            cur = {"start": steps[i], "peak_step": steps[i], "peak_loss": values[i], "baseline": med,
                   "threshold": thr, "last_i": i}
    if cur is not None:
        spikes.append(cur)
    for s in spikes:
        s["end"] = steps[s.pop("last_i")]
        s["excess"] = s["peak_loss"] - s["baseline"]
    return spikes


def precursor(steps, values, before: int, threshold: float, lookback: int = 200) -> int | None:
    """First step in (before - lookback, before] at which ``values`` reached ``threshold`` (None if never)."""
    for s, v in zip(steps, values):
        if before - lookback < s <= before and v >= threshold:
            return s
    return None


def growth(steps, values, upto: int, early: int = 10) -> float:
    """Ratio of a statistic just before ``upto`` to its median over the first ``early`` logged points."""
    pre = [v for s, v in zip(steps, values) if s <= upto]
    if len(pre) < 2:
        return float("nan")
    base = statistics.median(pre[:max(1, min(early, len(pre) // 2))])
    return pre[-1] / max(base, 1e-12)


def jump(steps, values, at: int, before: int = 20, after: int = 2) -> float:
    """Largest value in [at, at + after] divided by the median of the ``before`` logged points preceding ``at``."""
    pre = [v for s, v in zip(steps, values) if s < at][-before:]
    post = [v for s, v in zip(steps, values) if at <= s <= at + after]
    if not pre or not post:
        return float("nan")
    return max(post) / max(statistics.median(pre), 1e-12)


RULES = """Decision rules of lesson 07.5, applied to the first detected loss spike (start step s):
  logit growth   the max attention logit before s exceeds 20 and is >= 4x its median over the first 10 logged steps
  optimizer      otherwise, the update/weight ratio (max over matrices) in [s, s+2] is >= 4x its median over the
                 20 logged steps before s
  data           otherwise, if the loss is back within 2 thresholds of the pre-spike median within 20 steps
  unclear        anything else
With no spike: "logit growth (no spike)" if the logit rule holds over the whole run (at small scale growing logits
often stall the loss instead of spiking it), else "none". Spikes are found on the running median (width 5) of the
per-step losses.
These thresholds were set on the course's toy runs (lesson 07.5); treat them as a starting point, not a standard."""


def diagnose(metrics_rows: list[dict], stability_rows: list[dict], window: int = 20, k: float = 6.0,
             smooth_width: int = 5) -> dict:
    """Classify a run's first loss spike by :data:`RULES`. Returns the evidence with the verdict.

    Spikes are detected on the running median of the per-step losses (:func:`smooth`, width 5)."""
    st, loss = series(metrics_rows, "loss", "train")
    spikes = detect_spikes(st, smooth(loss, smooth_width) if smooth_width > 1 else loss, window, k)
    ls, lg = series(stability_rows, "max_logit")
    rs, rv = series(stability_rows, "ratio_max")
    ev = {"spikes": spikes}
    if not spikes:
        ev["max_logit_run"] = max(lg, default=float("nan"))
        g = growth(ls, lg, st[-1] if st else 0) if lg else float("nan")
        ev["logit_growth"] = g
        ev["verdict"] = "logit growth (no spike)" if (g >= 4 and ev["max_logit_run"] > 20) else "none"
        return ev
    s0 = spikes[0]
    start = s0["start"]
    ev["logit_growth"] = growth(ls, lg, start - 1) if lg else float("nan")
    ev["max_logit_before"] = max([v for s, v in zip(ls, lg) if s < start], default=float("nan"))
    ev["ratio_jump"] = jump(rs, rv, start) if rv else float("nan")
    after = [v for s, v in zip(st, loss) if s0["end"] < s <= s0["end"] + 20]
    ev["recovered"] = bool(after) and min(after) - s0["baseline"] < 2 * s0["threshold"]
    ev["verdict"] = classify_evidence(ev["logit_growth"], ev["max_logit_before"], ev["ratio_jump"], ev["recovered"])
    return ev


def classify_evidence(logit_growth: float, max_logit_before: float, ratio_jump: float, recovered: bool) -> str:
    if logit_growth >= 4 and max_logit_before > 20:
        return "logit growth"
    if ratio_jump >= 4:
        return "optimizer"
    if recovered:
        return "data"
    return "unclear"
