"""Run the course RL loop with a chosen objective, a control reward or a bounded-staleness sampler.

The loop is Module 12's :func:`frontierlab.posttrain.rl.train`, unchanged. This module only *wraps* it:

* the objective enters through the loop's ``policy_loss`` hook (:func:`objectives.as_hook`), and the
  objective's paired settings (advantage scale, aggregation, zero-variance handling) through the config;
* a **control reward** (``"random"`` or ``"format"``) is registered as an extra verifier name for training,
  while held-out evaluation is forced back to the strict verifier, so the control arm is scored exactly like
  every other arm. Every Stage D RL result carries such an arm: Shao et al. (2025) report large MATH-500
  gains on Qwen2.5-Math from random and format rewards, so "reward went up and accuracy went up" is not
  evidence that the verifier's signal did the work;
* :class:`BoundedStalenessSampler` replaces the loop's fixed-lag sampler with one whose lag varies per step
  (from an asynchronous schedule, lesson 14.3) but never exceeds the bound.

All three are applied inside :func:`patched_loop`, which restores the loop's module state on exit.

Control rewards (training only):

* ``random`` — Bernoulli(``p``) per response, independent of the response, from the global torch RNG (which
  the loop seeds and checkpoints, so a control run is reproducible and resumes exactly);
* ``format`` — 1 if the response finished with EOS and is digits only (the plain tag's format), whatever the
  number: a reward the warm-started policy already earns almost always.

The train-row ``pass`` of a control run is the control reward; its ``eval`` rows are strict accuracy.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from frontierlab.posttrain import arms as AR
from frontierlab.posttrain import rl
from frontierlab.posttrain import tasks
from frontierlab.rlscale.objectives import OBJECTIVES, as_hook

CONTROL_P = {"random": 0.5}


def random_verify(text: str, finished: bool, problem) -> bool:
    return bool(torch.rand(()) < CONTROL_P["random"])


def format_verify(text: str, finished: bool, problem) -> bool:
    return finished and bool(re.fullmatch(r"\d+", text))


CONTROLS = {"random": random_verify, "format": format_verify}


class BoundedStalenessSampler(rl.Sampler):
    """Samples with the policy from ``lag`` updates ago, ``lag = lags[n] <= bound`` for the n-th rollout.

    Keeps the last ``bound + 1`` snapshots (the loop pushes one after every update). ``lags`` comes from an
    asynchronous schedule (:func:`frontierlab.rlscale.asyncsim.simulate`); the loop's behaviour-policy
    log-probabilities are computed from the same snapshot, so the importance ratio stays honest. The lags used
    are recorded in ``self.used`` (a resumed run restarts the schedule; it is not bit-identical to a straight run).
    """

    lags: list[int] = []
    instances: list = []

    def __init__(self, policy, staleness: int, dtype: str):
        super().__init__(policy, staleness, dtype)
        self.n = 0
        self.used: list[int] = []
        type(self).instances.append(self)

    def behaviour_state(self):
        want = self.lags[self.n % len(self.lags)] if self.lags else self.k
        lag = min(want, self.k, len(self.snapshots) - 1)
        self.used.append(lag)
        self.n += 1
        return self.snapshots[-(lag + 1)]


@contextlib.contextmanager
def patched_loop(control: str | None = None, lags: list[int] | None = None):
    """Register control verifiers, force strict held-out evaluation, optionally swap in the bounded sampler."""
    for k, fn in CONTROLS.items():
        tasks.VERIFIERS.setdefault(k, fn)
    orig_eval, orig_sampler = rl.evaluate, rl.Sampler

    def strict_eval(policy, held, max_new, temperature, gen, verifier="strict"):
        return orig_eval(policy, held, max_new, temperature, gen, "strict" if verifier in CONTROLS else verifier)

    rl.evaluate = strict_eval
    cls = None
    if lags is not None:
        cls = type("BoundedStalenessSamplerRun", (BoundedStalenessSampler,), {"lags": list(lags), "instances": []})
        rl.Sampler = cls
    try:
        yield cls
    finally:
        rl.evaluate, rl.Sampler = orig_eval, orig_sampler


def objective_config(base: rl.RLConfig, objective: str, **over) -> rl.RLConfig:
    """The base config with the objective's paired loop settings, then ``over`` (seed, run, lr, ...)."""
    return replace(base, **{**OBJECTIVES[objective].loop, **over})


def train_objective(cfg: rl.RLConfig, objective: str, *, loss_fn=None, params: dict | None = None,
                    control: str | None = None, lags: list[int] | None = None, hooks: dict | None = None) -> dict:
    """Train one run. ``cfg`` should already carry the objective's loop settings (:func:`objective_config`)."""
    if control:
        cfg = replace(cfg, verifier=control)
    h = {"policy_loss": as_hook(objective, loss_fn, **(params or {}))}
    h.update(hooks or {})
    with patched_loop(control, lags) as cls:
        out = rl.train(cfg, h)
    if cls is not None and cls.instances:
        out["lags_used"] = cls.instances[-1].used
    return out


def run_arms(base: rl.RLConfig, arms: dict[str, dict], seeds: list[int], root: str | Path,
             loss_fns: dict | None = None) -> dict:
    """Like :func:`frontierlab.posttrain.arms.run_arms`, for objective arms.

    Each arm is ``{"objective": name, "params": {...}, "control": None | "random" | "format", "over": {...}}``;
    ``loss_fns`` maps objective names to replacement losses (the learner's). Finished runs are skipped.
    """
    root = Path(root)
    out = {}
    for name, arm in arms.items():
        out[name] = []
        for s in seeds:
            cfg = objective_config(base, arm["objective"], **arm.get("over", {}), seed=s, run=str(root / f"{name}-s{s}"))
            if not (Path(cfg.run) / "policy.pt").exists():
                print(f"== arm {name} seed {s}")
                train_objective(cfg, arm["objective"], loss_fn=(loss_fns or {}).get(arm["objective"]),
                                params=arm.get("params"), control=arm.get("control"), lags=arm.get("lags"))
            out[name].append(AR.summarise_run(cfg.run))
    return out


def initial_policy(path: str | Path, seed: int = 0) -> Path:
    """An untrained toy policy saved as a loop checkpoint (tests and smoke runs; no SFT needed)."""
    from frontierlab.model import LM
    from frontierlab.posttrain.sft import policy_config, save_policy
    path = Path(path)
    if not path.exists():
        torch.manual_seed(seed)
        save_policy(LM(policy_config()), path)
    return path


def lag_stats(lags) -> dict:
    a = np.asarray(lags, dtype=float)
    return {"mean": float(a.mean()) if a.size else 0.0, "max": int(a.max()) if a.size else 0,
            "hist": {int(k): int((a == k).sum()) for k in np.unique(a)} if a.size else {}}


__all__ = ["CONTROLS", "BoundedStalenessSampler", "patched_loop", "objective_config", "train_objective", "run_arms",
           "initial_policy", "lag_stats"]
