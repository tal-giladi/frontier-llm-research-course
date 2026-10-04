"""Parameters and FLOPs of a BlockLM, for equal-parameter and equal-FLOPs comparisons (Module 6).

``frontierlab.attention.accounting`` knows every attention kind but assumes a dense SwiGLU and one output
head. Module 6 adds parameters that are not used per token (unselected experts, unread Engram rows, the
PLE table), parameters used only in training (MTP modules) and work that is not in any weight matrix
(the n-stream residual). This module counts them; the proposed change to the shared accounting is in
``curriculum/inbox/module-06-shared-changes.md``.

* Parameters are counted on the real module tree, built on PyTorch's ``meta`` device (no memory), so the
  totals are exact by construction (a test checks them against a real model).
* FLOPs follow the course convention (multiply-add = 2; training = 3 × forward):

      forward/token = 2 · (active non-embedding parameters)        weights each token uses once
                    + 2·V·C · (1 + number of MTP predictions)      main head + one shared head per MTP prediction
                    + attention-score FLOPs (from attention.accounting) · (L + number of MTP blocks) / L
                    + n-stream residual work (``hyperconn.hc_flops_per_token_sublayer``, 2 per layer), whose
                      coefficient projections replace the 2·(their parameters) term

  "Active" excludes unselected routed experts (``(E - k)`` expert matrices per MoE layer), Engram tables
  and the PLE table (lookups cost 0 FLOPs, as BLT section 4.5 also assumes for embeddings), and, for
  MatFormer, the neurons beyond the expected width under uniform sampling. MTP modules are active in
  training and absent at inference (``training=False`` drops them).
"""

from __future__ import annotations

import torch

from frontierlab.attention import accounting as attn_acc
from frontierlab.blocks.hyperconn import hc_flops_per_token_sublayer
from frontierlab.blocks.model import BlockLM, block_settings
from frontierlab.model.config import ModelConfig


def _base(cfg: ModelConfig) -> ModelConfig:
    extra = {k: v for k, v in cfg.extra.items() if k != "blocks"}
    return cfg.with_(extra=extra)


def param_counts(cfg: ModelConfig) -> dict:
    """Exact parameter counts of ``BlockLM(cfg)``, split by role. Keys used by the loop: total, non_embedding,
    embedding. Also: mtp, hyper, engram_tables, ple_table, moe_inactive, active_non_embedding (inference)."""
    with torch.device("meta"):
        m = BlockLM(cfg)
    b = m.blocks
    seen, groups = set(), {"mtp": 0, "hyper": 0, "engram": 0, "engram_tables": 0, "ple": 0, "ple_table": 0}
    total = 0
    for name, p in m.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        if name.startswith("mtp."):
            groups["mtp"] += p.numel()
        elif name.startswith("model.hyper."):
            groups["hyper"] += p.numel()
        elif name.startswith("model.engram."):
            groups["engram"] += p.numel()
            if ".tables." in name:
                groups["engram_tables"] += p.numel()
        elif name.startswith("model.ple."):
            groups["ple"] += p.numel()
            if "embed_tokens_per_layer" in name:
                groups["ple_table"] += p.numel()
    emb = cfg.vocab_size * cfg.hidden_size
    moe_inactive = 0
    if b["ffn"] == "moe":
        mo = b["moe"]
        per_expert = 3 * cfg.hidden_size * mo["intermediate"]
        moe_inactive = (cfg.num_hidden_layers - mo["first_dense"]) * (mo["experts"] - mo["top_k"]) * per_expert
    non_emb = total - emb
    active = non_emb - groups["mtp"] - groups["engram_tables"] - groups["ple_table"] - moe_inactive
    return {"total": total, "embedding": emb, "non_embedding": non_emb, "moe_inactive": moe_inactive,
            "active_non_embedding": active, **groups}


def _attention_scores(cfg: ModelConfig, T: int) -> float:
    """Forward attention-score FLOPs per token of the whole stack (0 for weights), from attention.accounting.

    attention.accounting instantiates attention modules on the CPU, which draws from torch's global RNG; the
    loop calls this *after* restoring the checkpoint's RNG state, so without ``fork_rng`` a resumed run whose
    training uses the global RNG (the diffusion masks, dropout) would not be the same run (inbox item 4)."""
    with torch.random.fork_rng(devices=[]):
        return _attention_scores_inner(cfg, T)


def _attention_scores_inner(cfg: ModelConfig, T: int) -> float:
    base = _base(cfg)
    factor = 1.0
    if base.attention == "gqa-bidir":             # no causal mask: every query sees T keys, not T/2 on average
        base, factor = base.with_(attention="gqa"), 2.0
    return factor * (attn_acc.flops_per_token(base, T, training=False) - attn_acc._fwd_matmul(base))


def flops_per_token(cfg: ModelConfig, T: int, training: bool = True) -> float:
    """Average FLOPs per token at sequence length T (training = 3 × forward, MTP included; inference without MTP)."""
    b = block_settings(cfg)
    pc = param_counts(cfg)
    C, V, L = cfg.hidden_size, cfg.vocab_size, cfg.num_hidden_layers
    active = pc["active_non_embedding"] + (pc["mtp"] if training else 0)
    if b["ffn"] == "matformer":
        ws = b["matformer"]["widths"]
        mean_w = sum(ws) / len(ws) if training else cfg.intermediate_size
        active -= L * 3 * C * (cfg.intermediate_size - mean_w)
    n_mtp = int(b["mtp_depth"]) if (b["mtp"] != "none" and training) else 0
    head = 2 * V * C if cfg.tie_word_embeddings else 0        # an untied head is already in 2·N
    fwd = 2 * active + head + 2 * V * C * n_mtp
    fwd += _attention_scores(cfg, T) * (L + n_mtp) / L
    if b["residual"] != "plain":                  # its projection weights are already in 2·active: count them once
        fwd += 2 * L * hc_flops_per_token_sublayer(int(b["streams"]), C, b["residual"], int(b["sinkhorn_iters"]))
        fwd -= 2 * pc["hyper"]
    return 3 * fwd if training else fwd


def equal_flops_steps(base: ModelConfig, other: ModelConfig, base_steps: int, T: int) -> int:
    """Steps for ``other`` whose training FLOPs equal ``base`` trained ``base_steps`` steps (same tokens/step)."""
    return max(1, round(base_steps * flops_per_token(base, T) / flops_per_token(other, T)))


def breakdown(cfg: ModelConfig, T: int) -> dict:
    """Training FLOPs per token split into the terms above (for the lessons' cost tables)."""
    b = block_settings(cfg)
    pc = param_counts(cfg)
    C, V, L = cfg.hidden_size, cfg.vocab_size, cfg.num_hidden_layers
    n_mtp = int(b["mtp_depth"]) if b["mtp"] != "none" else 0
    attn = _attention_scores(cfg, T)
    hc = 2 * L * hc_flops_per_token_sublayer(int(b["streams"]), C, b["residual"], int(b["sinkhorn_iters"])) \
        if b["residual"] != "plain" else 0
    return {"weights": 3 * 2 * (pc["active_non_embedding"] - pc["hyper"]), "mtp_weights": 3 * 2 * pc["mtp"],
            "head": 3 * 2 * V * C, "mtp_heads": 3 * 2 * V * C * n_mtp, "attention": 3 * attn,
            "mtp_attention": 3 * attn * n_mtp / L, "residual_streams": 3 * hc, "total": flops_per_token(cfg, T)}
