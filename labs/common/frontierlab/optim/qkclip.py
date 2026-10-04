"""Per-head maximum attention logits during training, and QK-Clip (Kimi K2's MuonClip), lesson 07.2.

**The statistic.** For head h of a layer, Kimi K2 (arXiv 2507.20534, section 2.1) defines the maximum logit
over a training batch as

    S_max^h = max over the batch and over allowed (i, j) of  q_i^h · k_j^h / sqrt(d)

(the scale is the one the attention actually uses; MLA uses 1/sqrt(d_n + d_r)). :class:`LogitMonitor`
computes it with a forward pre-hook on each attention module: it takes the module's input, recomputes
that head's queries and keys with the module's own projections (no gradient), and takes the masked maximum,
batch element by batch element so the (H, T, T) logits of only one sequence exist at a time. The forward
itself is unchanged. Only training forwards count (``module.training``), so evaluation does not clip.
With gradient accumulation the maximum is taken over all micro-batches of the step.

**QK-Clip.** After the optimizer step, every head with S_max^h > tau is rescaled so that, on the same
batch, its maximum logit would be exactly tau: gamma_h = min(1, tau / S_max^h), applied to the weights
(Kimi K2 used tau = 100). Where gamma goes:

* MHA (``num_key_value_heads == num_attention_heads``) without QK-norm: W_q rows of head h and W_k rows of
  head h each times sqrt(gamma_h) (Kimi K2's rule for head-specific q and k).
* GQA (several query heads share one key head): W_q rows of head h times gamma_h; the shared key is left
  untouched, as Kimi K2 leaves MLA's shared rotary key untouched. Course choice (INFERENCE from the MLA
  rule): scaling the shared key would also shrink the logits of the other heads in its group.
* MLA (``frontierlab.attention.mla``): the head-specific non-rotary parts q^C and k^C times sqrt(gamma_h)
  (rows of ``q_proj``/``q_b_proj`` and the W_UK rows of ``kv_b_proj``), the head-specific rotary query
  q^R times gamma_h, the shared rotary key k^R (from ``kv_a_proj_with_mqa``) untouched — Kimi K2 section 2.1.

Logits are linear in each of these factors (RoPE is a rotation; the RMSNorm on MLA's latent sits *before*
``kv_b_proj``), so after the rescale the recomputed S_max^h equals tau for clipped heads (tested).

With QK-norm (Baseline-0's default) QK-Clip does nothing useful: the RMSNorm after ``q_proj`` and ``k_proj``
undoes any rescaling of their rows, and the norm gain is shared by all heads. :class:`QKClip` refuses such a
model; QK-norm *is* the alternative fix (DeepSeek-V4 applies RMSNorm to queries and KV entries instead of
QK-Clip, arXiv 2606.19348 sections 2.3.3 and 2.4).
"""

from __future__ import annotations

import math

import torch

from frontierlab.attention.base import apply_rope
from frontierlab.attention.ops import band_mask


def _is_mla(mod) -> bool:
    return hasattr(mod, "kv_b_proj") and hasattr(mod, "latents")


def _qk(mod, x, positions):
    """Queries (B, H, T, dq), keys (B, H or KV, T, dq) and the logit scale, exactly as the module scores them."""
    if _is_mla(mod):
        cos, sin = mod.rope(positions)
        q_nope, q_rope = mod.queries(x, cos, sin)
        c, k_rope = mod.latents(x, cos, sin)
        B, S, _ = c.shape
        k_nope = mod.kv_b_proj(c).view(B, S, mod.H, mod.d_n + mod.d_v).transpose(1, 2)[..., :mod.d_n]
        k = torch.cat((k_nope, k_rope.expand(B, mod.H, S, mod.d_r)), dim=-1)
        return torch.cat((q_nope, q_rope), -1), k, mod.scale
    if hasattr(mod, "project"):                        # Module 3 windowed/sink/gated kinds
        q, k, _ = mod.project(x, positions)
        return q, k, q.shape[-1] ** -0.5
    B, T, _ = x.shape                                  # Baseline-0 GQA and kinds with its attributes
    q = mod.q_norm(mod.q_proj(x).view(B, T, mod.H, mod.hd)).transpose(1, 2)
    k = mod.k_norm(mod.k_proj(x).view(B, T, mod.KV, mod.hd)).transpose(1, 2)
    cos, sin = mod.rope(positions)
    return apply_rope(q, cos, sin), apply_rope(k, cos, sin), mod.hd ** -0.5


@torch.no_grad()
def head_max_logits(mod, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """S_max per query head (H,), float32 or wider, for one attention module and its input x (B, T, C)."""
    window = getattr(mod, "window", None)
    mask = band_mask(positions, positions, window)
    out = None
    for b in range(x.shape[0]):
        q, k, scale = _qk(mod, x[b:b + 1], positions)
        H = q.shape[1]
        if k.shape[1] != H:
            k = k.repeat_interleave(H // k.shape[1], dim=1)
        work = torch.promote_types(q.dtype, torch.float32)                 # float64 models stay float64
        lg = (q.to(work) @ k.to(work).transpose(-1, -2)) * scale           # (1, H, T, T)
        m = lg.masked_fill(~mask, float("-inf")).amax(dim=(0, 2, 3))
        out = m if out is None else torch.maximum(out, m)
    return out


def attention_modules(model) -> list:
    return [layer.self_attn for layer in model.model.layers]


class LogitMonitor:
    """Collects S_max per layer and head over the training forwards of one optimizer step.

    ``active`` switches collection on (QK-Clip keeps it on; the stability logger turns it on for logged steps).
    ``take()`` returns {layer: (H,) tensor} and resets.
    """

    def __init__(self, model):
        self.model = model
        self.active = False
        self.smax: dict[int, torch.Tensor] = {}
        self.handles = []
        for i, mod in enumerate(attention_modules(model)):
            self.handles.append(mod.register_forward_pre_hook(self._hook(i)))

    def _hook(self, i):
        def hook(mod, args):
            if not (self.active and mod.training and torch.is_grad_enabled()):
                return
            x, positions = args[0], args[1]
            cache = args[2] if len(args) > 2 else None
            if cache is not None:
                return
            m = head_max_logits(mod, x.detach(), positions)
            self.smax[i] = m if i not in self.smax else torch.maximum(self.smax[i], m)
        return hook

    def take(self) -> dict:
        out, self.smax = self.smax, {}
        return out

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def _rows(weight: torch.Tensor, start: int, stop: int, factor: float):
    weight[start:stop].mul_(factor)


@torch.no_grad()
def clip_module(mod, gamma: torch.Tensor) -> None:
    """Rescale one attention module's weights by per-head gamma (H,) as described in the module docstring."""
    if _is_mla(mod):
        dq = mod.d_n + mod.d_r
        wq = mod.q_b_proj.weight if mod.r_q else mod.q_proj.weight          # (H*dq, C or r_q)
        wkv = mod.kv_b_proj.weight                                          # (H*(d_n+d_v), d_c)
        for h, g in enumerate(gamma.tolist()):
            if g >= 1.0:
                continue
            s = math.sqrt(g)
            _rows(wq, h * dq, h * dq + mod.d_n, s)                          # q^C
            _rows(wq, h * dq + mod.d_n, (h + 1) * dq, g)                    # q^R (head-specific rotary)
            _rows(wkv, h * (mod.d_n + mod.d_v), h * (mod.d_n + mod.d_v) + mod.d_n, s)   # k^C = W_UK c
        return                                                              # k^R (shared) untouched
    H, KV, d = mod.H, mod.KV, mod.hd
    for h, g in enumerate(gamma.tolist()):
        if g >= 1.0:
            continue
        if KV == H:
            s = math.sqrt(g)
            _rows(mod.q_proj.weight, h * d, (h + 1) * d, s)
            _rows(mod.k_proj.weight, h * d, (h + 1) * d, s)
        else:
            _rows(mod.q_proj.weight, h * d, (h + 1) * d, g)


def has_qk_norm(mod) -> bool:
    return not _is_mla(mod) and not isinstance(getattr(mod, "q_norm", None), torch.nn.Identity)


class QKClip:
    """QK-Clip as an optimizer post-step hook. ``QKClip(model, tau).attach(opt)``.

    ``last`` holds, after each step, {"max_logit": float, "clipped_heads": int, "min_gamma": float}.
    """

    def __init__(self, model, tau: float = 100.0, monitor: LogitMonitor | None = None, allow_qk_norm: bool = False):
        mods = attention_modules(model)
        if not allow_qk_norm and any(has_qk_norm(m) for m in mods):
            raise ValueError("QK-Clip rescales W_q/W_k rows, which QK-norm undoes; use qk_norm=False "
                             "(or MLA), or keep QK-norm as the fix instead of QK-Clip")
        self.model, self.tau, self.mods = model, float(tau), mods
        self.monitor = monitor or LogitMonitor(model)
        self.monitor.active = True
        self.last: dict = {}

    def attach(self, opt):
        opt.post_step_hooks.append(self)
        return self

    def __call__(self, opt=None):
        smax = self.monitor.take()
        clipped, min_gamma, top = 0, 1.0, 0.0
        layers = [round(float(smax[i].max()), 4) for i in sorted(smax)]
        for i, s in smax.items():
            gamma = torch.clamp(self.tau / s.clamp_min(1e-12), max=1.0)
            clipped += int((gamma < 1).sum())
            min_gamma = min(min_gamma, float(gamma.min()))
            top = max(top, float(s.max()))
            clip_module(self.mods[i], gamma)
        self.last = {"max_logit": top, "clipped_heads": clipped, "min_gamma": min_gamma, "layers": layers}
