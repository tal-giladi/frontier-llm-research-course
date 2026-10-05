"""Micro-anneals: score a candidate dataset by a short annealing run from a mid-training checkpoint (lesson 10.4).

OLMo 2 (section 4.4.2) evaluates math sources with *microanneals*: take roughly equal amounts of the
candidate data and of the general mix, train that 50/50 mixture "as if it were an annealing run" (the
learning rate driven linearly to zero over the short run), and compare the result with the checkpoint
and with other candidates. They ran 19 of them, 130B tokens in total, from a 7B model that had
completed pretraining, with effects visible after fewer than 10B tokens.

This module builds the arms of such a comparison and runs them through ``frontierlab.datax.train``:

* :func:`anneal_spec` — the 50/50 (or ``frac``) mixture of a base mix and one candidate source;
* :func:`run_microanneals` — for every candidate and the **control** (the base mix alone, same tokens,
  same schedule: without it, any gain could be the anneal itself), one run per seed with
  ``--init-from <stable checkpoint> --anneal``; the seed changes the data order, the weights start the
  same. Returns the run folders by arm and seed, ready for :mod:`frontierlab.datax.arms` scoring.

The course anneals start from the *weights* of the stable-phase checkpoint with a fresh optimizer and a
short warmup (``--init-from``), as Module 4 did for context extension; a run that also carries the
optimizer state over (``frontierlab.optim.train --branch-from``) is the closer match to a real anneal and
is a stated limit of the lab.
"""

from __future__ import annotations

import copy
from pathlib import Path

from frontierlab.datax import arms
from frontierlab.datax.mixture import MixtureSpec, SourceRef


def anneal_spec(base: MixtureSpec, candidate: SourceRef | None, frac: float = 0.5, name: str | None = None) -> MixtureSpec:
    """``base`` scaled to (1 − frac) plus ``candidate`` at ``frac``; ``candidate=None`` is the control (base alone)."""
    s = copy.deepcopy(base)
    if candidate is None:
        s.name = name or f"{base.name}-control"
        return s
    tot = sum(r.weight for r in s.sources)
    srcs = [SourceRef(r.name, r.root, (1 - frac) * r.weight / tot, r.split) for r in s.sources]
    srcs.append(SourceRef(candidate.name, candidate.root, frac, candidate.split))
    return MixtureSpec(srcs, seed=s.seed, block=s.block, name=name or f"{base.name}+{candidate.name}")


def run_microanneals(out: Path, stable_ckpt: Path, base: MixtureSpec, candidates: dict[str, SourceRef], *,
                     preset: str, steps: int, batch: int, seq: int, lr: float, warmup: int, seeds: list[int],
                     frac: float = 0.5, device: str = "cpu") -> dict:
    arms_ = {"control": anneal_spec(base, None)}
    arms_.update({k: anneal_spec(base, c, frac) for k, c in candidates.items()})
    runs: dict = {}
    for name, spec in arms_.items():
        runs[name] = {}
        for s in seeds:
            runs[name][s] = arms.train_arm(Path(out) / f"anneal-{name}-s{s}", spec, preset=preset, steps=steps,
                                           batch=batch, seq=seq, lr=lr, warmup=warmup, seed=s, device=device,
                                           extra=["--init-from", str(stable_ckpt), "--anneal"],
                                           question=f"micro-anneal: does {name} improve its target domain? (10.4)")
    return runs
