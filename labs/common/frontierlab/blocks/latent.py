"""Latent reasoning loops (lesson 06.7, extension): Coconut's continuous thoughts and a recurrent-depth core.

**Coconut** (Hao et al., arXiv 2412.06769, section 3): in "latent mode" the model's last hidden state is fed
back as the next input embedding instead of being decoded into a token. Here the fed-back vector is the
final hidden state after the model's final RMSNorm (GPT-2's last hidden state, which Coconut used, is also
after its final layer norm). Each continuous thought is one more forward step over the cache, so k thoughts
cost k decode steps; training needs k + 1 sequential forward passes per example (section 3: "We perform
n + 1 forward passes when n latent thoughts are scheduled") and a curriculum that replaces language
reasoning steps by thoughts stage by stage.

**Recurrent depth** (Geiping et al., arXiv 2502.05171, section 3): prelude P (l_P layers) embeds, a core
block R (l_R layers) is iterated r times on a state s, the coda C (l_C layers) decodes:

    e = P(x);  s_0 ~ N(0, sigma² I);  s_i = R(A [s_{i-1}; e]),  i = 1..r;  p = C(s_r)

with an adapter A: R^{2h} -> R^h. Training samples r per step from a log-normal Poisson distribution
(tau ~ N(log(r_bar) − sigma²/2, sigma), r ~ Poisson(e^tau) + 1, section 3.3) and backpropagates through only
the last k iterations (k = 8 in the paper), so memory does not grow with r. Effective depth is
l_P + l_R · r + l_C (the paper's (2, 4, 2) model at r = 32 unrolls to 132 layers).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from frontierlab.layers.rmsnorm import RMSNorm
from frontierlab.model.lm import Block


@torch.no_grad()
def continuous_thoughts(model, idx: torch.Tensor, k: int):
    """Run ``idx`` (B, T) through the cache, then k continuous-thought steps. Returns (logits after the last
    thought (B, V), thoughts (B, k, C)). Works with any BlockLM without Engram (thoughts have no token id)."""
    cache = model.new_cache()
    out = model(idx, cache=cache, return_hidden=True)
    thoughts = []
    h = model.model.norm(out.hidden["h"][:, -1:])
    for _ in range(k):
        thoughts.append(h)
        out = model(inputs_embeds=h, cache=cache, return_hidden=True)
        h = model.model.norm(out.hidden["h"][:, -1:])
    return out.logits[:, -1], torch.cat(thoughts, 1) if thoughts else h[:, :0]


def sample_recurrence(mean_r: float, sigma: float = 0.5, generator: torch.Generator | None = None) -> int:
    """Log-normal Poisson draw of the number of core iterations (recurrent-depth paper, section 3.3)."""
    tau = torch.normal(math.log(mean_r) - 0.5 * sigma ** 2, sigma, (1,), generator=generator)
    return int(torch.poisson(tau.exp(), generator=generator).item()) + 1


class RecurrentDepthLM(nn.Module):
    """A small (l_P, l_R, l_C) recurrent-depth model built from the course's blocks (no decode cache)."""

    def __init__(self, cfg, prelude: int = 1, core: int = 2, coda: int = 1, init_sigma: float = 1.0):
        super().__init__()
        C = cfg.hidden_size
        self.config, self.init_sigma = cfg, init_sigma
        self.embed_tokens = nn.Embedding(cfg.vocab_size, C)
        self.prelude = nn.ModuleList(Block(cfg, i) for i in range(prelude))
        self.adapter = nn.Linear(2 * C, C, bias=False)
        self.core = nn.ModuleList(Block(cfg, prelude + i) for i in range(core))
        self.coda = nn.ModuleList(Block(cfg, prelude + core + i) for i in range(coda))
        self.norm = RMSNorm(C, eps=cfg.rms_norm_eps)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=cfg.initializer_range)

    def effective_depth(self, r: int) -> int:
        return len(self.prelude) + len(self.core) * r + len(self.coda)

    def forward(self, idx: torch.Tensor, r: int, backprop_last: int | None = None, seed: int = 0) -> torch.Tensor:
        """Logits (B, T, V). ``backprop_last`` = k: iterations before the last k run without autograd."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        e = self.embed_tokens(idx)
        for blk in self.prelude:
            e = blk(e, pos)
        g = torch.Generator().manual_seed(seed)
        s = (torch.randn(e.shape, generator=g) * self.init_sigma).to(e)
        k = r if backprop_last is None else min(r, backprop_last)
        for i in range(r):
            if i < r - k:
                with torch.no_grad():
                    s = self._core(s, e, pos)
            else:
                s = self._core(s, e, pos)
        for blk in self.coda:
            s = blk(s, pos)
        return self.norm(s) @ self.embed_tokens.weight.T                   # tied head

    def _core(self, s, e, pos):
        s = self.adapter(torch.cat([s, e], -1))
        for blk in self.core:
            s = blk(s, pos)
        return s
