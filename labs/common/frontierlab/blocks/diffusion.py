"""A tiny masked-diffusion language model objective, LLaDA-style (lesson 06.7, extension).

LLaDA (Nie et al., arXiv 2502.09992, section 2.1) replaces next-token prediction with a *mask predictor*:
a Transformer without a causal mask that sees a partly masked sequence and predicts every masked token.

    forward process:  each token of x_0 is replaced by [M] independently with probability t, t ~ U[0, 1]
    loss (Eq. 3):     L = - E_{t, x_0, x_t} [ (1/t) · sum_i 1[x_t^i = M] · log p_theta(x_0^i | x_t) ]
    bound (Eq. 4):    - E[log p_theta(x_0)] <= L       (the 1/t weight is what makes it a likelihood bound;
                                                         MaskGIT's objective lacks it, LLaDA section 2.1)

The per-sequence sum is divided by the sequence length L here so the number is "nats per token" and
comparable in scale with an autoregressive model's loss (an upper bound on it, not the same quantity).
For evaluation LLaDA uses a lower-variance equivalent (Eq. 6): draw l uniformly from {1..L}, mask
exactly l tokens uniformly at random, weight by L/l. Both estimators have the same expectation; the test
suite checks it exactly by enumerating every mask on a 3-token sequence.

Sampling (LLaDA section 2.4) starts from a fully masked response and, over ``steps`` steps, predicts all
masked tokens and keeps the most confident ones ("low-confidence remasking": the least confident
predictions are masked again). LLaDA cannot use a KV cache (section 2.2: it "is incompatible with KV
caching"), because every position attends to every other and masked positions change at every step.

The attention kind ``"gqa-bidir"`` is Baseline-0's GQA without the causal mask (registered here; the line
for ``frontierlab/attention/__init__.py`` is in the Module 6 inbox). ``DiffusionLM`` is a BlockLM whose
last vocabulary id is the mask token.
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F

from frontierlab.attention.base import apply_rope, register
from frontierlab.attention.gqa import GQAttention
from frontierlab.blocks.model import BlockLM, BlockLMOutput


@register("gqa-bidir")
class BidirectionalGQAttention(GQAttention):
    """Baseline-0's attention with no mask: every position attends to every position. No decode cache."""

    def forward(self, x, positions, cache=None):
        if cache is not None:
            raise ValueError("bidirectional attention has no incremental decode cache")
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.H, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.KV, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.KV, self.hd).transpose(1, 2)
        cos, sin = self.rope(positions)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, enable_gqa=self.H != self.KV)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.hd))


def mask_tokens(x0: torch.Tensor, t: torch.Tensor, mask_id: int, generator: torch.Generator | None = None):
    """Forward process: x0 (B, L) long, t (B,) in (0, 1]. Returns (x_t, masked bool (B, L))."""
    u = torch.rand(x0.shape, generator=generator, device="cpu").to(x0.device)
    masked = u < t[:, None]
    return torch.where(masked, torch.full_like(x0, mask_id), x0), masked


def diffusion_loss_terms(logits: torch.Tensor, x0: torch.Tensor, masked: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Per-position terms of Eq. 3, divided by L: (1/t) · 1[masked] · (−log p(x0 | x_t)) / L, shape (B, L)."""
    lg = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), x0.reshape(-1), reduction="none").view(x0.shape)
    return masked.to(ce.dtype) * ce / t[:, None].to(ce.dtype) / x0.shape[1]


class DiffusionLM(BlockLM):
    """Mask predictor over Baseline-0's blocks. ``cfg.attention`` must be "gqa-bidir"; mask id = vocab_size − 1.

    ``forward(x0, labels=x0)`` draws t and the mask (torch's global RNG in training — saved by the loop, so
    exact resume holds; a fixed-seed generator in eval mode, so evaluation is deterministic) and returns the
    Eq. 3 Monte-Carlo loss; ``per_token_loss`` (B, L) holds the per-position terms."""

    eval_seed = 1234

    def __init__(self, cfg):
        if cfg.attention != "gqa-bidir":
            raise ValueError("DiffusionLM needs attention='gqa-bidir'")
        super().__init__(cfg)
        self.mask_id = cfg.vocab_size - 1

    def forward(self, idx=None, labels=None, cache=None, reduction: str = "mean", inputs_embeds=None,
                return_hidden: bool = False):
        if labels is None:
            return super().forward(idx, cache=cache, inputs_embeds=inputs_embeds, return_hidden=return_hidden)
        gen = None if self.training else torch.Generator().manual_seed(self.eval_seed)
        B = labels.shape[0]
        t = torch.rand(B, generator=gen).clamp_min(1e-3).to(labels.device)         # t ~ U(0, 1]
        xt, masked = mask_tokens(labels, t, self.mask_id, gen)
        logits = super().forward(xt).logits
        terms = diffusion_loss_terms(logits, labels, masked, t)                    # (B, L)
        per_seq = terms.sum(1)
        loss = per_seq.mean() if reduction == "mean" else per_seq.sum()
        self.last_stats = {"main_loss": float(loss.detach()), "mask_rate": float(masked.float().mean())}
        return BlockLMOutput(logits, loss, terms * labels.shape[1], extras=self.last_stats)


@torch.no_grad()
def nll_bound(model: DiffusionLM, x0: torch.Tensor, samples: int = 8, seed: int = 0) -> torch.Tensor:
    """Eq. 6 estimate of the per-token NLL upper bound for each sequence of x0 (B, L): returns (B,) nats/token.

    Each sample: l ~ U{1..L}; mask l positions chosen uniformly without replacement; term (L/l)·sum of CE over
    the masked positions; the bound is the mean over samples divided by L."""
    was = model.training
    model.eval()
    g = torch.Generator().manual_seed(seed)
    B, L = x0.shape
    acc = torch.zeros(B, dtype=torch.float64)
    for _ in range(samples):
        l = torch.randint(1, L + 1, (B,), generator=g)
        order = torch.rand(B, L, generator=g).argsort(1)
        masked = order < l[:, None]                                  # l random positions per row
        xt = torch.where(masked.to(x0.device), torch.full_like(x0, model.mask_id), x0)
        logits = BlockLM.forward(model, xt).logits
        ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), x0.reshape(-1),
                             reduction="none").view(B, L).cpu().double()
        acc += (L / l.double()) * (ce * masked.double()).sum(1)
    model.train(was)
    return acc / samples / L


def exact_bound(logp_fn, x0: list[int], mask_id: int, form: str = "eq3") -> float:
    """The exact expectation of Eq. 3 (``form="eq3"``) or Eq. 6 (``"eq6"``) for one short sequence, by
    enumerating every mask set S (2^L of them). ``logp_fn(xt)`` returns (L, V) log-probabilities.

    Eq. 3: weight of S with |S| = s is the integral over t of (1/t) t^s (1-t)^(L-s) = (s-1)!(L-s)!/L!.
    Eq. 6: P(l = s) · P(S | l = s) · L/s = (1/L) · 1/C(L, s) · L/s, the same number (tested)."""
    L = len(x0)
    total = 0.0
    for s in range(1, L + 1):
        for S in itertools.combinations(range(L), s):
            xt = [mask_id if i in S else x0[i] for i in range(L)]
            with torch.no_grad():
                lp = logp_fn(xt)
            nll = -sum(float(lp[i, x0[i]]) for i in S)
            if form == "eq3":
                w = math.factorial(s - 1) * math.factorial(L - s) / math.factorial(L)
            else:
                w = (1 / L) * (1 / math.comb(L, s)) * (L / s)
            total += w * nll
    return total


@torch.no_grad()
def sample(model: DiffusionLM, prompt: torch.Tensor, gen_len: int, steps: int) -> torch.Tensor:
    """Low-confidence remasking sampler (LLaDA section 2.4), greedy: prompt (1, P) -> (1, P + gen_len).

    Start with gen_len mask tokens; at each of ``steps`` steps predict every masked position and commit the
    ``ceil(remaining / steps_left)`` most confident predictions; the rest stay masked."""
    was = model.training
    model.eval()
    x = torch.cat([prompt, torch.full((1, gen_len), model.mask_id, device=prompt.device)], 1)
    for step in range(steps):
        masked = x == model.mask_id
        remaining = int(masked.sum())
        if remaining == 0:
            break
        logits = BlockLM.forward(model, x).logits[0]
        logits[:, model.mask_id] = float("-inf")                    # never predict the mask token itself
        prob, pred = logits.softmax(-1).max(-1)
        prob = torch.where(masked[0], prob, torch.full_like(prob, -1.0))
        k = math.ceil(remaining / (steps - step))
        commit = prob.topk(k).indices
        x[0, commit] = pred[commit]
    model.train(was)
    return x
