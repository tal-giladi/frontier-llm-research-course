"""Reference solution for lab 16.3 — multi-turn agentic RL."""

from __future__ import annotations

import torch


def loss_mask(action_mask: torch.Tensor, obs_mask: torch.Tensor, loss_on: str = "actions") -> torch.Tensor:
    if loss_on == "actions":
        return action_mask.clone()
    if loss_on == "all":
        return (action_mask + obs_mask).clamp(max=1.0)
    raise ValueError(loss_on)


def turn_returns(turn_rewards: list[float], final: float, gamma: float = 1.0) -> list[float]:
    r = list(turn_rewards) + [final]
    out, acc = [], 0.0
    for x in reversed(r):
        acc = x + gamma * acc
        out.append(acc)
    return list(reversed(out))


def spread_over_tokens(turn_adv: list[float], turn_index: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(turn_index.shape)
    for t, a in enumerate(turn_adv):
        out[turn_index == t] = a
    return out


def context_cost(prompt: int, actions: list[int], observations: list[int], policy: str = "full",
                 window: int = 1) -> dict:
    ctx = []
    for t in range(len(actions)):
        prev = list(zip(actions[:t], observations[:t]))
        if policy == "window":
            prev = prev[-window:] if window > 0 else []
        ctx.append(prompt + sum(a + o for a, o in prev))
    return {"prefill_tokens": sum(ctx), "peak_context": max(c + a for c, a in zip(ctx, actions)) if ctx else prompt}
