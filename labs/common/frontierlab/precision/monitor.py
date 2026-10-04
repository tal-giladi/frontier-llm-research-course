"""Quantisation statistics during training: ``<run>/precision.jsonl`` (lessons 08.2–08.3).

The Module 7 stability log (``frontierlab.optim.stability``) watches logits and update sizes. A low-precision run
adds three questions per quantised tensor: how much does it lose (relative error), how many nonzero values
become zero (underflow), and how many hit the format's maximum (saturation)? :class:`PrecisionLogger` asks the
:class:`~frontierlab.precision.linear.QLinear` layers to compute these on every ``every``-th optimizer step (in
float64, on the tensors the GEMMs actually quantised) and writes one row per logged step:

    step                         optimizer step just taken
    act_rel_err, weight_rel_err, grad_rel_err      mean over layers of ||q(x) - x|| / ||x||
    act_underflow, grad_underflow                  mean fraction of nonzero values flushed to zero
    act_saturated, weight_saturated, grad_saturated
    worst_grad_layer, worst_grad_rel_err           the layer whose gradient lost the most

Logging changes no numbers: the statistics are computed on copies, and no random numbers are drawn.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

from frontierlab.precision.linear import QLinear


def _opt_steps(opt) -> int:
    if hasattr(opt, "steps_taken"):
        return int(opt.steps_taken)
    for st in opt.state.values():
        if "step" in st:
            return int(st["step"])
    return 0


class PrecisionLogger:
    def __init__(self, model, path, every: int = 10):
        self.model, self.every = model, max(1, int(every))
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")
        self.layers = {n: m for n, m in model.named_modules() if isinstance(m, QLinear)}
        self._armed = True                      # the first step of a (re)started run is always logged
        self._h1 = model.register_forward_pre_hook(self._arm)
        self._h2 = register_optimizer_step_post_hook(self._after_step)

    def _arm(self, module, args):
        if module.training:
            for m in self.layers.values():
                m.collect = self._armed

    def _after_step(self, opt, args, kwargs):
        step = _opt_steps(opt)
        if self._armed and self.layers:
            row = {"split": "precision", "step": step}
            for role in ("act", "weight", "grad"):
                vals = {n: m.last_stats[role] for n, m in self.layers.items() if role in m.last_stats}
                if not vals:
                    continue
                for key in ("rel_err", "underflow", "saturated"):
                    row[f"{role}_{key}"] = statistics.mean(v[key] for v in vals.values())
                if role == "grad":
                    worst = max(vals, key=lambda n: vals[n]["rel_err"])
                    row["worst_grad_layer"], row["worst_grad_rel_err"] = worst, vals[worst]["rel_err"]
            self._f.write(json.dumps(row) + "\n")
            self._f.flush()
        for m in self.layers.values():
            m.collect, m.last_stats = False, {}
        self._armed = (step + 1) % self.every == 0

    def close(self):
        self._h1.remove()
        self._h2.remove()
        self._f.close()
