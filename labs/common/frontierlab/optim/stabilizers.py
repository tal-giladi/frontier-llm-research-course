"""Loss and logit stabilizers that need a model change: z-loss, soft-capping, QKV clamping (lesson 07.5).

* **z-loss** (PaLM, arXiv 2204.02311 section 5: "an auxiliary loss of z_loss = 10^-4 · log^2 Z to encourage
  the softmax normalizer log(Z) to be close to 0"; OLMo 2 uses weight 10^-5, arXiv 2501.00656 Table 1).
  Per token, log Z = logsumexp of the output logits; the term added to the training loss is
  ``z_loss · mean(log Z ^ 2)``. It is added only in training; ``per_token_loss`` stays the plain
  cross-entropy, so evaluation is comparable across arms.
* **final-logit soft-capping** (Gemma 2, arXiv 2408.00118 section 2: logits <- cap · tanh(logits / cap),
  cap 30 on the final layer and 50 in attention).
* **attention-logit soft-capping and QKV clamping**: the attention kind ``"gqa-softcap"`` is Baseline-0's GQA
  with ``extra["attn_softcap"]`` (tanh cap on the scaled scores, Gemma 2) and/or ``extra["clip_qkv"]``
  (clamp the q, k and v projections to [-c, c] before QK-norm and RoPE, as Hugging Face ``OlmoAttention``
  does for OLMo-1.7-7B's ``"clip_qkv": 8.0``). It computes the softmax explicitly (no SDPA), so it costs
  O(T·S) memory per head; fine at course scale.

:class:`OptLM` is ``frontierlab.model.LM`` with these settings read from ``cfg.extra`` (``z_loss``,
``final_softcap``). Its state dict is the LM's, so checkpoints load into either class; evaluation of a
soft-capped or µP model must use :func:`frontierlab.optim.train.load_model`, which rebuilds the same class.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from frontierlab.attention.base import apply_rope, causal_mask, register
from frontierlab.attention.gqa import GQAttention
from frontierlab.model.lm import LM, LMOutput


class OptLM(LM):
    """LM with optional z-loss and final-logit soft-cap; records log Z statistics when ``track_logz`` is set."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.track_logz = False
        self.last_logz: dict | None = None

    def forward(self, idx, labels=None, cache=None, reduction: str = "mean") -> LMOutput:
        out = super().forward(idx, None, cache)
        logits = out.logits
        cap = self.config.extra.get("final_softcap")
        if cap:
            logits = cap * torch.tanh(logits / cap)
        if labels is None:
            return LMOutput(logits)
        B, T = idx.shape
        flat = logits[:, :-1].reshape(-1, logits.size(-1))
        tok = F.cross_entropy(flat, labels[:, 1:].reshape(-1), reduction="none").view(B, T - 1)
        loss = tok.mean() if reduction == "mean" else tok.sum()
        z = float(self.config.extra.get("z_loss", 0.0) or 0.0)
        if self.training and (z or self.track_logz):
            lse = torch.logsumexp(flat, dim=-1)
            if z:
                loss = loss + z * lse.pow(2).mean()
            if self.track_logz:
                d = lse.detach()
                self.last_logz = {"logz_mean": d.mean().item(), "logz_max": d.abs().max().item(),
                                  "logit_max": logits.detach().abs().max().item()}
        return LMOutput(logits, loss, tok)


@register("gqa-softcap")
class SoftcapGQAttention(GQAttention):
    """GQA (Baseline-0) with optional attention-logit soft-cap and QKV clamping (module docstring)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.attn_softcap = cfg.extra.get("attn_softcap")
        self.clip_qkv = cfg.extra.get("clip_qkv")
        self.window = None

    def project(self, x, positions):
        B, T, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.clip_qkv:
            c = float(self.clip_qkv)
            q, k, v = q.clamp(-c, c), k.clamp(-c, c), v.clamp(-c, c)
        q = self.q_norm(q.view(B, T, self.H, self.hd)).transpose(1, 2)
        k = self.k_norm(k.view(B, T, self.KV, self.hd)).transpose(1, 2)
        v = v.view(B, T, self.KV, self.hd).transpose(1, 2)
        cos, sin = self.rope(positions)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def forward(self, x, positions, cache=None):
        B, T, _ = x.shape
        q, k, v = self.project(x, positions)
        k_pos = positions
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat((cache["k"], k), 2), torch.cat((cache["v"], v), 2)
                k_pos = torch.cat((cache["pos"], positions))
            cache["k"], cache["v"], cache["pos"] = k, v, k_pos
        rep = self.H // self.KV
        kk = k.repeat_interleave(rep, 1) if rep > 1 else k
        vv = v.repeat_interleave(rep, 1) if rep > 1 else v
        s = (q @ kk.transpose(-1, -2)) * self.hd ** -0.5                     # (B, H, T, S)
        if self.attn_softcap:
            s = self.attn_softcap * torch.tanh(s / self.attn_softcap)
        s = s.masked_fill(~causal_mask(positions, k_pos), float("-inf"))
        work = torch.float32 if s.dtype in (torch.float16, torch.bfloat16) else s.dtype
        p = torch.softmax(s.to(work), -1).to(vv.dtype)
        y = p @ vv
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.hd))
