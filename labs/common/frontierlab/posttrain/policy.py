"""Sampling and log-probabilities for a causal LM policy (any model with the Baseline-0 interface:
``model(idx, cache=...) -> .logits`` and ``model.new_cache()``).

Shapes (B sequences, P prompt tokens, R = max new tokens):

* ``Rollout.tokens``      (B, P + R) int64  prompt then response, PAD after EOS
* ``Rollout.response``    (B, R)     int64  the response part
* ``Rollout.mask``        (B, R)     float32  1 on response tokens up to and including EOS, else 0
* ``Rollout.sampler_logp``(B, R)     float32  log-probability of each sampled token under the *sampler*
  (the inference engine's numbers; 0 where the mask is 0)
* ``Rollout.finished``    (B,)       bool   the response produced EOS within R tokens (False = truncated)

The trainer recomputes log-probabilities with :func:`token_logprobs` in one teacher-forced forward;
the sampler's numbers come from cached decoding, possibly in another precision. They agree only up to
rounding, which is why lesson 12.3 treats them as two different policies.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from frontierlab.posttrain.tokenizer import EOS, PAD


@dataclass
class Rollout:
    tokens: torch.Tensor
    response: torch.Tensor
    mask: torch.Tensor
    sampler_logp: torch.Tensor
    finished: torch.Tensor
    prompt_len: int

    @property
    def lengths(self) -> torch.Tensor:
        return self.mask.sum(-1)


def response_mask(response: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(mask, finished) for responses (B, R): mask is 1 up to and including the first EOS, then 0.

    A response with no EOS is truncated: every one of its R tokens is in the mask.
    """
    is_eos = response == EOS
    # number of EOS strictly before each position; a token is kept while none came before it
    before = torch.cumsum(is_eos.long(), dim=-1) - is_eos.long()
    mask = (before == 0).float()
    return mask, is_eos.any(-1)


@torch.no_grad()
def sample(model, prompts: torch.Tensor, max_new: int, temperature: float = 1.0,
           generator: torch.Generator | None = None) -> Rollout:
    """Sample ``max_new`` tokens per prompt with the KV cache. After a row's EOS its tokens are PAD."""
    was_training = model.training
    model.eval()
    B, P = prompts.shape
    cache = model.new_cache()
    logits = model(prompts, cache=cache).logits[:, -1].float()
    out = torch.full((B, max_new), PAD, dtype=torch.long, device=prompts.device)
    logp = torch.zeros((B, max_new), dtype=torch.float32, device=prompts.device)
    done = torch.zeros(B, dtype=torch.bool, device=prompts.device)
    for t in range(max_new):
        lp = torch.log_softmax(logits / temperature, dim=-1)
        nxt = torch.multinomial(lp.exp(), 1, generator=generator).squeeze(-1)
        nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
        out[:, t] = nxt
        logp[:, t] = torch.where(done, torch.zeros_like(logp[:, t]), lp.gather(-1, nxt[:, None]).squeeze(-1))
        done = done | (nxt == EOS)
        if bool(done.all()) or t == max_new - 1:
            break
        logits = model(nxt[:, None], cache=cache).logits[:, -1].float()
    model.train(was_training)
    mask, finished = response_mask(out)
    return Rollout(torch.cat([prompts, out], 1), out, mask, logp * mask, finished, P)


def token_logprobs(model, tokens: torch.Tensor, prompt_len: int, temperature: float = 1.0,
                   with_entropy: bool = False):
    """Teacher-forced log-probabilities of the response tokens ``tokens[:, prompt_len:]``.

    Returns ``logp`` (B, R) float32 and, if asked, the entropy of the policy at each response position
    (B, R) in nats. Logits are cast to float32 before the softmax (ScaleRL's "FP32 logits" point).
    """
    logits = model(tokens).logits[:, prompt_len - 1:-1].float() / temperature       # (B, R, V)
    lp_all = torch.log_softmax(logits, dim=-1)
    logp = lp_all.gather(-1, tokens[:, prompt_len:, None]).squeeze(-1)
    if not with_entropy:
        return logp
    ent = -(lp_all.exp() * lp_all).sum(-1)
    return logp, ent


@torch.no_grad()
def exact_token_kl(model, ref, tokens: torch.Tensor, prompt_len: int, temperature: float = 1.0) -> torch.Tensor:
    """(B, R) exact KL(pi(.|context) || ref(.|context)) at every response position, summed over the whole
    vocabulary (possible for a 35-token vocabulary; on a real model it costs a full logits tensor)."""
    lp = torch.log_softmax(model(tokens).logits[:, prompt_len - 1:-1].float() / temperature, -1)
    lr = torch.log_softmax(ref(tokens).logits[:, prompt_len - 1:-1].float() / temperature, -1)
    return (lp.exp() * (lp - lr)).sum(-1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(1.0)
