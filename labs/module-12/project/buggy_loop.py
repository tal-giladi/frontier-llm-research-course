"""Module 12 project, debugging task: a colleague's "improved" RL loop that trains worse than it should.

    python labs/module-12/project/buggy_loop.py             # their loop, 120 steps, 2 seeds (about 2 minutes)
    python labs/module-12/project/buggy_loop.py --fixed     # the course loop, same settings (after your diagnosis)
    LOOP_HOOKS=buggy pytest labs/module-12/project          # the detail tests against their pieces

Their message: "I refactored three pieces of the loop: grouping rewards before the advantage, the loss
mask, and where the old log-probabilities come from (so we don't have to keep a copy of the sampler).
Training reward goes up a bit, so it works, but it's slower than your runs and the clip fraction looks
odd. Can you check before I launch the big run?"

The three pieces are below, exactly as they plugged them into ``frontierlab.posttrain.rl.train``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from frontierlab.posttrain import advantages as A
from frontierlab.posttrain import arms as AR
from frontierlab.posttrain import rl
from frontierlab.posttrain.policy import token_logprobs
from frontierlab.posttrain.sft import ensure_sft, load_policy

ROOT = Path("runs/m12/project-debug")


def grouped_advantages(R: torch.Tensor, baseline: str, scale: str) -> torch.Tensor:
    """Group the rewards per prompt and normalise within each group."""
    P, G = R.shape
    groups = R.reshape(-1).view(G, P).t()
    return A.group_advantages(groups, baseline, scale)


def response_loss_mask(mask: torch.Tensor, rollout) -> torch.Tensor:
    """Every generated position counts; the sampler pads after EOS, so padded rows are all the same length."""
    return torch.ones_like(mask) * (mask.sum(-1, keepdim=True) > 0).float()


_START = {}


def old_logprobs(default, policy, rollout, cfg):
    """Recompute the old log-probs instead of storing the sampler's (saves a model copy)."""
    if cfg.init not in _START:
        _START[cfg.init] = load_policy(cfg.init).eval()
    with torch.no_grad():
        return token_logprobs(_START[cfg.init], rollout.tokens, rollout.prompt_len, cfg.temperature)


BUGGY_HOOKS = {"advantages": grouped_advantages, "loss_mask": response_loss_mask, "old_logp": old_logprobs}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixed", action="store_true")
    a = ap.parse_args(argv)
    sft = ensure_sft("runs/m12/sft")
    base = rl.RLConfig(init=str(sft), steps=120, eval_every=40)
    hooks = {} if a.fixed else BUGGY_HOOKS
    name = "fixed" if a.fixed else "buggy"
    original = rl.train
    AR.train = lambda cfg: original(cfg, hooks)          # noqa: E731
    res = AR.run_arms(replace(base), {name: {}}, [0, 1], ROOT)
    for metric in ("final_sampled", "train_pass_auc", "clip_frac", "ratio_max", "len", "entropy", "kl_exact"):
        print(AR.table(res, metric))
    (ROOT / f"{name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
