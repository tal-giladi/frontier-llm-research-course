"""Train an EAGLE-style draft head on a frozen target (lesson 15.2).

The head is one :class:`frontierlab.blocks.mtp.DeepSeekModule` — RMSNorm both inputs, concatenate, project
2C -> C, one transformer block, a final RMSNorm — read through the *target's* frozen embedding and output head.
That is DeepSeek-V3's MTP module (Eq. 21–23) trained after the fact instead of jointly, which is what an EAGLE
draft head is (Li et al. 2024: autoregression at the feature level, input = the target's top-layer feature and
the embedding of the token one step ahead). Entry i reads (h_i, Emb(x_{i+1})) and predicts x_{i+2}.

Loss (EAGLE section 3.2: feature regression plus token classification):

    L = SmoothL1(g_i, h_{i+1}) + w_cls · CE(p_{i+2}, p̂_{i+2}),      w_cls = 0.1 in EAGLE

where p_{i+2} = softmax(OutHead(h_{i+1})) is the *target's* next-token distribution (a soft label: the head
is distilled from the target, not trained on the data's tokens) and p̂_{i+2} = softmax(OutHead(g_i)).

The regression term makes the head's output feature g_i usable as the stand-in for h_{i+1} when it drafts a
second token. EAGLE-3 drops the regression, predicts tokens directly from a fusion of low, middle and high
layer features, and trains with "training-time test" (feeding its own predictions during training); this
course head does neither, and the lesson says so where it compares numbers.

Only the head's parameters are trained; the target runs in ``torch.no_grad``.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from frontierlab.blocks.mtp import DeepSeekModule
from frontierlab.ttc.speculative import forward_hidden


def new_head(cfg, seed: int = 0) -> DeepSeekModule:
    torch.manual_seed(seed)
    head = DeepSeekModule(cfg, cfg.num_hidden_layers)
    for m in head.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.normal_(m.weight, std=cfg.initializer_range)
    return head


def head_loss(head, target, x: torch.Tensor, w_cls: float = 0.1) -> tuple[torch.Tensor, dict]:
    """x (B, T) token windows. Returns (loss, parts) for entries i = 0 .. T-3."""
    with torch.no_grad():
        _, h = forward_hidden(target, x)                     # (B, T, C), pre-final-norm
    T = x.shape[1]
    hp = head.combine(h[:, :T - 2], target.model.embed_tokens(x[:, 1:T - 1]))
    g = head.block(hp, torch.arange(T - 2, device=x.device))
    reg = F.smooth_l1_loss(g, h[:, 1:T - 1])
    logits = target.lm_head(head.norm(g)).float()
    with torch.no_grad():
        p_t = torch.softmax(target.lm_head(target.model.norm(h[:, 1:T - 1])).float(), -1)
    ce = -(p_t * torch.log_softmax(logits, -1)).sum(-1).mean()
    with torch.no_grad():
        top1 = (logits.argmax(-1) == p_t.argmax(-1)).float().mean()      # agreement with the target's greedy choice
    return reg + w_cls * ce, {"reg": float(reg.detach()), "ce": float(ce.detach()), "top1": float(top1)}


def train_head(target, data, steps: int = 300, batch: int = 16, seq: int = 128, lr: float = 2e-3,
               w_cls: float = 0.1, seed: int = 0, log_every: int = 50, device: str = "cpu"):
    """Train a head on random windows of ``data`` (a ``frontierlab.data.loader.TokenData``)."""
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)
    head = new_head(target.config, seed).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    gen = torch.Generator().manual_seed(seed)
    hist, t0 = [], time.perf_counter()
    for step in range(1, steps + 1):
        for g_ in opt.param_groups:
            g_["lr"] = lr * min(1.0, step / 20)
        x = data.batch(batch, seq, gen, device)
        loss, parts = head_loss(head, target, x, w_cls)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps:
            hist.append({"step": step, "loss": float(loss), **parts, "seconds": round(time.perf_counter() - t0, 1)})
    for p in target.parameters():
        p.requires_grad_(True)
    return head.eval(), hist
