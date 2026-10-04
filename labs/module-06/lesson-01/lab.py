"""Lab 06.1 — multi-token prediction. Fill in the TODOs; run `pytest labs/module-06/lesson-01` to check."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: F401  (you will need it)

from frontierlab.attention import accounting as attn_acc  # noqa: F401  (TODO 5)


def mtp_chain(model, h0: torch.Tensor, idx: torch.Tensor) -> list[torch.Tensor]:
    """DeepSeek-V3's sequential MTP modules (V3 section 2.2, Eqs. 21-23).

    ``model`` is a BlockLM built with mtp="deepseek"; its modules are ``model.mtp.layers[k-1]``, each with
    ``hnorm``, ``enorm`` (RMSNorm), ``eh_proj`` (Linear 2C -> C), ``block`` (a transformer block called as
    ``block(x, positions)``) and ``norm``. h0 (B, T, C) is the main model's final hidden state before its final
    norm; idx (B, T) the input tokens. For depth k = 1..D, over the S = T - k positions i = 0..S-1:

        h'_i = eh_proj( concat( hnorm(h^{k-1}_i), enorm(Emb(idx[i + k])) ) )
        h^k  = block(h', positions 0..S-1)
        logits^k = model.lm_head(norm(h^k))                (B, S, V)

    The embedding is ``model.model.embed_tokens``. Return the list of logits, one per depth.
    """
    raise NotImplementedError("TODO 1: the sequential MTP chain")


def mtp_loss(depth_logits: list[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    """Mean over depths of the cross-entropy of depth k's logits at position i against labels[i + k + 1]
    (V3 Eqs. 24-25, without lambda). Use only positions whose target exists."""
    raise NotImplementedError("TODO 2: shift the targets by k + 1 and average over depths")


def lambda_schedule(step: int, steps: int) -> float:
    """DeepSeek-V3's MTP weight (section 4.2): 0.3 for the first 10T of 14.8T tokens, then 0.1. ``step`` of ``steps``."""
    raise NotImplementedError("TODO 3: the lambda schedule")


def accepted_prefix(drafts: list[int], verify: list[int]) -> int:
    """Greedy speculative decoding: drafts d_1..d_m are accepted while d_j equals the main head's greedy choice
    verify[j] at that position; the first mismatch rejects it and everything after it. Return the count."""
    raise NotImplementedError("TODO 4: count the accepted drafts")


def mtp_extra_flops(cfg, T: int, depth: int) -> float:
    """Training FLOPs per token that ``depth`` DeepSeek MTP modules add to a model with config ``cfg`` at length T.

    Course convention (multiply-add = 2, training = 3 × forward). Per module, forward: 2 × its parameters
    (one transformer block, eh_proj with 2C·C weights, three RMSNorm gains of C), plus 2·V·C for the shared
    output head, plus one layer's attention-score FLOPs. ``attn_acc.param_counts(cfg)`` gives
    ``attention_per_layer`` (a list) and ``mlp_per_layer``; a block also has two norm gains (2C). One layer's
    attention-score FLOPs: (``attn_acc.flops_per_token(cfg, T, training=False)`` − ``attn_acc._fwd_matmul(cfg)``) / L.
    """
    raise NotImplementedError("TODO 5: the FLOPs MTP adds")
