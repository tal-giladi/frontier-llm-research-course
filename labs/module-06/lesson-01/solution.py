"""Reference solution for lab 06.1 (multi-token prediction)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from frontierlab.attention import accounting as attn_acc


def mtp_chain(model, h0: torch.Tensor, idx: torch.Tensor) -> list[torch.Tensor]:
    """DeepSeek-V3's sequential MTP (section 2.2): logits of every depth, list of (B, T-k, V)."""
    out, h = [], h0
    T = idx.shape[1]
    for k, mod in enumerate(model.mtp.layers, start=1):
        S = T - k
        if S <= 0:
            break
        emb_ahead = model.model.embed_tokens(idx[:, k:k + S])                       # Emb(t_{i+k})
        hp = mod.eh_proj(torch.cat([mod.hnorm(h[:, :S]), mod.enorm(emb_ahead)], -1))  # M_k [RMSNorm(h); RMSNorm(e)]
        h = mod.block(hp, torch.arange(S, device=idx.device))                     # TRM_k
        out.append(model.lm_head(mod.norm(h)))                                      # shared output head
    return out


def mtp_loss(depth_logits: list[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    """(1/D) sum_k CE(depth k at position i, labels[i + k + 1])  (V3 Eqs. 24-25 without lambda)."""
    losses = []
    for k, lg in enumerate(depth_logits, start=1):
        S = min(lg.shape[1], labels.shape[1] - k - 1)
        losses.append(F.cross_entropy(lg[:, :S].reshape(-1, lg.size(-1)), labels[:, k + 1:k + 1 + S].reshape(-1)))
    return torch.stack(losses).mean()


def lambda_schedule(step: int, steps: int) -> float:
    """0.3 for the first 10/14.8 of the run, then 0.1 (DeepSeek-V3 section 4.2: 10T of 14.8T tokens)."""
    return 0.3 if step / steps < 10.0 / 14.8 else 0.1


def accepted_prefix(drafts: list[int], verify: list[int]) -> int:
    """Number of drafts accepted by greedy verification: the longest prefix with drafts[j] == verify[j]."""
    n = 0
    for d, v in zip(drafts, verify):
        if d != v:
            break
        n += 1
    return n


def mtp_extra_flops(cfg, T: int, depth: int) -> float:
    """Training FLOPs per token that ``depth`` DeepSeek MTP modules add to ``cfg`` (course convention).

    Per module: 2 × (block + eh_proj (2C·C) + three norm gains (3C)) for its weights, 2·V·C for the shared head,
    and one layer's attention-score FLOPs; training = 3 × forward."""
    C, V, L = cfg.hidden_size, cfg.vocab_size, cfg.num_hidden_layers
    pc = attn_acc.param_counts(cfg)
    block = pc["attention_per_layer"][-1] + pc["mlp_per_layer"] + 2 * C
    module = block + 2 * C * C + 3 * C
    attn_layer = (attn_acc.flops_per_token(cfg, T, training=False) - attn_acc._fwd_matmul(cfg)) / L
    return 3 * depth * (2 * module + 2 * V * C + attn_layer)
