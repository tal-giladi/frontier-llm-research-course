# Copied from the Mixture-of-Experts Engineering course (labs/common/moelab: routing/topk.py,
# balance/losses.py and the Experts class of model/lm.py, as of commit 7b9d27f), so frontierlab stands
# alone and imports nothing from moelab. Changes: imports, a shared expert, a module that plugs into the
# frontierlab block (``MoEFFN``) and a dense reference for the tests. Routing, balancing, fast expert
# execution and expert parallelism are taught in that course; here the layer is a component.
"""A fine-grained Mixture-of-Experts FFN for the course model (Module 6, Lineage-F).

Layout and parameter names follow Hugging Face Qwen3-MoE / Qwen2-MoE, so a checkpoint maps by name:

    mlp.gate.weight                     (E, C)       router, one score per expert
    mlp.experts.gate_up_proj            (E, 2I, C)   rows [0, I) gate, rows [I, 2I) up, for every expert
    mlp.experts.down_proj               (E, C, I)
    mlp.shared_expert.{gate,up,down}_proj           an always-on SwiGLU of width I_s (DeepSeekMoE-style)

Forward for N = B·T tokens (shapes in the docstrings below):

    probs  = softmax(x W_r^T)  in fp32                  (N, E)
    w, idx = top-k(probs); w = w / sum(w)               (N, k)   gate weights sum to 1 per token
    y      = sum_j w_j · Expert_{idx_j}(x)  +  Shared(x)

"Fine-grained" (DeepSeekMoE, arXiv 2401.06066): many small experts (I = intermediate_size / m) with a
larger top-k, plus shared experts, instead of a few large ones. The auxiliary balance loss is the
Switch form E · sum_i f_i P_i (equals 1 when routing is uniform); ``BlockLM`` adds
``moe_aux_coef`` times its mean over MoE layers to the training loss. Aux-loss-free balancing
(DeepSeek-V3 section 2.1.2) is taught in the MoE course and not repeated here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RouterOutput:
    """What a router hands to the expert layer (and to the metrics).

    Shapes for N tokens, E experts, top-k = k:
        logits:      (N, E)  raw router scores, in the input dtype
        probs:       (N, E)  fp32 probabilities over all experts (used by aux losses and metrics)
        topk_idx:    (N, k)  long, selected expert ids (distinct within a row)
        topk_weight: (N, k)  gate weights applied to the selected experts' outputs
    """

    logits: torch.Tensor
    probs: torch.Tensor
    topk_idx: torch.Tensor
    topk_weight: torch.Tensor


class TopKRouter(nn.Module):
    """Linear router + softmax + top-k, with optional renormalisation of the kept gates (moelab, unchanged)."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int, norm_topk_prob: bool = True):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.num_experts, self.top_k, self.norm_topk_prob = num_experts, top_k, norm_topk_prob
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))

    def forward(self, x: torch.Tensor) -> RouterOutput:
        x = x.reshape(-1, self.weight.shape[1])                      # (N, C)
        logits = F.linear(x, self.weight)                             # (N, E)
        probs = torch.softmax(logits, dim=-1, dtype=torch.promote_types(x.dtype, torch.float32))
        w, idx = torch.topk(probs, self.top_k, dim=-1)                # (N, k), (N, k)
        if self.norm_topk_prob:
            w = w / w.sum(dim=-1, keepdim=True)
        return RouterOutput(logits, probs, idx, w.to(x.dtype))


class Experts(nn.Module):
    """E SwiGLU experts stored as two fused tensors (the Transformers layout), executed by a loop over experts."""

    def __init__(self, num_experts: int, hidden_size: int, intermediate: int):
        super().__init__()
        self.num_experts, self.intermediate = num_experts, intermediate
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * intermediate, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate))

    def expert(self, e: int, x: torch.Tensor) -> torch.Tensor:
        """Run expert ``e`` on tokens ``x`` of shape (n, C)."""
        gate, up = F.linear(x, self.gate_up_proj[e]).chunk(2, dim=-1)                 # (n, I) each
        return F.linear(F.silu(gate) * up, self.down_proj[e])                          # (n, C)

    def forward(self, x: torch.Tensor, r: RouterOutput) -> torch.Tensor:
        out = torch.zeros_like(x)                                                      # (N, C)
        for e in range(self.num_experts):
            tok, slot = torch.where(r.topk_idx == e)          # tokens that picked e, and in which slot
            if tok.numel() == 0:
                continue
            y = self.expert(e, x[tok]) * r.topk_weight[tok, slot, None]
            out = out.index_add(0, tok, y.to(out.dtype))
        return out


def switch_aux(r: RouterOutput, num_experts: int) -> torch.Tensor:
    """E · sum_i f_i P_i; f_i = share of routing slots sent to expert i (no gradient), P_i = mean router prob."""
    f = torch.bincount(r.topk_idx.reshape(-1), minlength=num_experts).to(r.probs.dtype) / r.topk_idx.numel()
    return num_experts * torch.sum(f * r.probs.mean(0))


class SharedExpert(nn.Module):
    """The always-on expert, with Qwen3 MLP names (gate_proj, up_proj, down_proj)."""

    def __init__(self, hidden_size: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEFFN(nn.Module):
    """Drop-in replacement for the block's SwiGLU: ``y = MoEFFN(x)`` with x, y of shape (B, T, C).

    After each forward, ``last_router`` holds the :class:`RouterOutput` and ``last_aux`` the Switch loss
    (BlockLM collects them). Settings: E experts of width I, top-k, ``shared`` experts of width I (0 = none).
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int, intermediate: int, shared: int = 1,
                 norm_topk_prob: bool = True):
        super().__init__()
        self.gate = TopKRouter(hidden_size, num_experts, top_k, norm_topk_prob)
        self.experts = Experts(num_experts, hidden_size, intermediate)
        self.shared_expert = SharedExpert(hidden_size, shared * intermediate) if shared else None
        self.last_router: RouterOutput | None = None
        self.last_aux: torch.Tensor | None = None

    @torch.no_grad()
    def reset_parameters(self, std: float):
        self.gate.weight.normal_(0.0, std)
        self.experts.gate_up_proj.normal_(0.0, std)
        self.experts.down_proj.normal_(0.0, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        flat = x.reshape(-1, C)
        r = self.gate(flat)
        y = self.experts(flat, r)
        if self.shared_expert is not None:
            y = y + self.shared_expert(flat)
        self.last_aux = switch_aux(r, self.gate.num_experts)
        self.last_router = RouterOutput(*(t.detach() for t in (r.logits, r.probs, r.topk_idx, r.topk_weight)))
        return y.view(B, T, C)


def dense_reference(moe: MoEFFN, x: torch.Tensor) -> torch.Tensor:
    """The same function computed the slow, obvious way: every expert on every token, then the gate-weighted
    sum of the top-k (a dense loop over tokens). Used by the tests to check the sparse dispatch."""
    B, T, C = x.shape
    flat = x.reshape(-1, C)
    probs = torch.softmax(flat @ moe.gate.weight.T, -1)
    w, idx = probs.topk(moe.gate.top_k, -1)
    if moe.gate.norm_topk_prob:
        w = w / w.sum(-1, keepdim=True)
    out = torch.zeros_like(flat)
    for n in range(flat.shape[0]):
        for j in range(moe.gate.top_k):
            out[n] += w[n, j] * moe.experts.expert(int(idx[n, j]), flat[n:n + 1])[0]
    if moe.shared_expert is not None:
        out = out + moe.shared_expert(flat)
    return out.view(B, T, C)
