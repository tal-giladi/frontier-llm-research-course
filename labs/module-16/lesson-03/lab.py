"""Lab 16.3 — multi-turn agentic RL. Fill in the TODOs; run `pytest labs/module-16/lesson-03` to check.
``multiturn_lab.py`` checks your functions on real rollouts and runs the credit and masking arms.
"""

from __future__ import annotations

import torch


def loss_mask(action_mask: torch.Tensor, obs_mask: torch.Tensor, loss_on: str = "actions") -> torch.Tensor:
    """(B, R) float mask of the tokens that enter the policy loss.

    ``loss_on="actions"``: exactly the policy-sampled tokens. ``loss_on="all"`` (the bug this lab measures):
    sampled tokens and inserted observation tokens. Any other value raises ValueError.
    """
    raise NotImplementedError("TODO 1: the loss mask")


def turn_returns(turn_rewards: list[float], final: float, gamma: float = 1.0) -> list[float]:
    """Return-to-go for one episode. ``turn_rewards[t]`` is the shaped reward of turn t (one per tool call);
    the final answer is one more turn whose reward is ``final``. G_t = r_t + gamma * G_{t+1}, so the result has
    len(turn_rewards) + 1 entries and the last one equals ``final``."""
    raise NotImplementedError("TODO 2: returns over turns")


def spread_over_tokens(turn_adv: list[float], turn_index: torch.Tensor) -> torch.Tensor:
    """(R,) float32: token r gets ``turn_adv[turn_index[r]]``; tokens with turn index -1 (padding) or a turn
    with no advantage get 0."""
    raise NotImplementedError("TODO 3: per-turn advantages onto tokens")


def context_cost(prompt: int, actions: list[int], observations: list[int], policy: str = "full",
                 window: int = 1) -> dict:
    """Token counts of one episode under a context policy (lengths in tokens).

    Action t is generated from a context of ``prompt`` plus the earlier (action, observation) pairs: all of
    them for ``policy="full"``, only the last ``window`` pairs for ``policy="window"``. Return
    {"prefill_tokens": sum over actions of the context each one starts from, "peak_context": the largest
    context plus that action's own length}. With no actions: prefill 0 and peak ``prompt``.
    """
    raise NotImplementedError("TODO 4: the cost of a context policy")
