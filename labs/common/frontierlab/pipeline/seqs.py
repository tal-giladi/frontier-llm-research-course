"""Sequences of (prompt, response) token ids: padding, response masks, sequence log-probabilities and the
supervised loss that every stage of a post-training pipeline reuses (Module 13).

Conventions (any causal LM with ``model(ids).logits``; no attention mask is needed because padding only
comes *after* the last real token of a row, and a causal model never looks to the right):

* ``Example(prompt, response)``: tuples of token ids; ``response`` ends with EOS if it finished.
* :func:`pad` -> ``ids`` (B, T) int64 = prompt + response + PAD..., and ``mask`` (B, T) float32 that is 1
  at every position holding a *response* token (a target), 0 on prompt and PAD positions.
* The token at position t is predicted by the logits at t - 1, so the functions below compare
  ``logits[:, :-1]`` with ``ids[:, 1:]`` under ``mask[:, 1:]``.

Prompts may have different lengths: each row's response starts at its own offset.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from frontierlab.pipeline.compute import Ledger, n_params


@dataclass(frozen=True)
class Example:
    prompt: tuple
    response: tuple

    @staticmethod
    def of(prompt, response) -> "Example":
        return Example(tuple(int(t) for t in prompt), tuple(int(t) for t in response))

    def __len__(self):
        return len(self.prompt) + len(self.response)


def pad(examples: list, pad_id: int = 0, max_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    T = max_len or max(len(e) for e in examples)
    ids = torch.full((len(examples), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(examples), T), dtype=torch.float32)
    for i, e in enumerate(examples):
        row = list(e.prompt) + list(e.response)
        if len(row) > T:
            raise ValueError("example longer than max_len")
        ids[i, :len(row)] = torch.tensor(row, dtype=torch.long)
        mask[i, len(e.prompt):len(row)] = 1.0
    return ids, mask


def token_logps(model, ids: torch.Tensor) -> torch.Tensor:
    """(B, T-1) log-probability of ids[:, 1:] given the prefix (float32, or float64 for a float64 model)."""
    logits = model(ids).logits[:, :-1]
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    return torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).squeeze(-1)


def sequence_logps(model, ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(log pi(response | prompt) summed over the response tokens (B,), response length (B,))."""
    m = mask[:, 1:]
    return (token_logps(model, ids) * m).sum(-1), m.sum(-1)


def sft_loss(model, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy over response tokens only (prompt and PAD are given, not predicted)."""
    m = mask[:, 1:]
    return -(token_logps(model, ids) * m).sum() / m.sum().clamp_min(1.0)


def batches(n: int, batch: int, steps: int, gen: torch.Generator):
    """``steps`` index tensors of up to ``batch`` items, without replacement within each pass over n items."""
    perm, pos = torch.randperm(n, generator=gen), 0
    for _ in range(steps):
        if pos + batch > n:
            perm, pos = torch.randperm(n, generator=gen), 0
        yield perm[pos:pos + min(batch, n)]
        pos += batch


def train_sft(model, examples: list, steps: int, batch: int = 64, lr: float = 1e-3, seed: int = 0,
              warmup: int = 20, grad_clip: float = 1.0, ledger: Ledger | None = None, log=None,
              log_every: int = 50, pad_id: int = 0) -> dict:
    """Supervised fine-tuning on ``examples`` (AdamW, linear warm-up, then constant). The ledger is charged
    6 N per non-PAD token: prompt tokens go through the forward and backward pass although they carry no loss."""
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    N = n_params(model)
    model.train()
    t0, hist = time.perf_counter(), []
    for step, idx in enumerate(batches(len(examples), batch, steps, gen), start=1):
        ids, mask = pad([examples[i] for i in idx.tolist()], pad_id)
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, step / max(1, warmup))
        loss = sft_loss(model, ids, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if ledger is not None:
            ledger.train("student_train", N, float((ids != pad_id).sum()))
        if step % log_every == 0 or step == steps:
            row = {"stage": "sft", "step": step, "loss": float(loss.detach()), "seconds": round(time.perf_counter() - t0, 2)}
            hist.append(row)
            if log is not None:
                log.log(**row)
    model.eval()
    return {"steps": steps, "final_loss": hist[-1]["loss"] if hist else math.nan, "history": hist,
            "seconds": round(time.perf_counter() - t0, 2)}
