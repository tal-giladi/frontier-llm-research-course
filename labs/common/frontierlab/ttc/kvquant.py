"""KV-cache and weight quantisation for serving (lesson 15.4, extension). Emulation: numerics only, no speed.

**Asymmetric group quantisation.** A group of values x is stored as b-bit integers q in {0, ..., 2^b - 1} with a
scale s = (max - min) / (2^b - 1) and a zero point z = min:  x ≈ s·q + z.  The scale and zero point are kept in
16 bits each, so a group of G elements costs b + 32/G bits per element. Symmetric formats (Module 8,
``frontierlab.precision.quant``) waste half the grid when a group is not centred on zero; KV entries often
are not, which is why KIVI uses zero points.

**KIVI's layout** (Liu et al. 2024, arXiv 2402.02750, sections 3–4): *keys per channel* — each channel's values
over G consecutive tokens share a scale, because a few key channels carry large outliers across all tokens;
*values per token* — each token's d values share a scale. Only complete groups are quantised; the most recent
R tokens (a residual window) stay in full precision, so the newest tokens, which attention weights most,
are exact. The paper uses G = 32 and R = 128.

:class:`QuantKVAttention` registers ``"gqa-kvq"``: Baseline-0's GQA with that cache. The emulation keeps the
full-precision cache and, at every forward with a cache, attends with ``kivi_qdq`` applied to it. Groups are
aligned to absolute positions, so a token's quantised value never changes once its group has left the
residual window: the result is what a real quantise-once cache would give. Evaluate with one-token decode
steps (:func:`decode_nll`), the serving situation; a full-sequence forward without a cache is unquantised.

Settings (``cfg.extra``): ``kv_bits`` [4], ``kv_group`` [32], ``kv_residual`` [128], ``kv_key_axis``
["channel" (KIVI) | "token"], ``kv_quant_values`` [True].

**Weights.** :func:`quantize_weights_` rounds every linear inside the blocks to b-bit groups (round to nearest,
RTN), asymmetric with zero points (the usual "w4a16 g128" baseline that GPTQ and AWQ improve on) or symmetric
through Module 8's ``precision.ptq_``. The embedding and the tied head are left in full precision.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from frontierlab.attention.base import apply_rope, causal_mask, register
from frontierlab.attention.gqa import GQAttention


# --------------------------------------------------------------------------------------------- the quantiser

def asym_qdq(x: torch.Tensor, bits: int, dim: int, group: int) -> torch.Tensor:
    """Quantise-dequantise ``x`` in groups of ``group`` consecutive elements along ``dim`` (asymmetric, zero
    point, round to nearest). The size of ``dim`` must be a multiple of ``group`` (callers pass whole groups)."""
    if x.shape[dim] % group:
        raise ValueError(f"size {x.shape[dim]} along dim {dim} is not a multiple of group {group}")
    xd = x.movedim(dim, -1)
    shp = xd.shape
    g = xd.reshape(*shp[:-1], shp[-1] // group, group)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    levels = 2 ** bits - 1
    s = (hi - lo) / levels
    safe = torch.where(s > 0, s, torch.ones_like(s))
    q = torch.clamp(torch.round((g - lo) / safe), 0, levels)
    out = torch.where(s > 0, q * s + lo, g)                 # a constant group is exact
    return out.reshape(shp).movedim(-1, dim)


def bits_per_element(bits: int, group: int, meta_bits: int = 32) -> float:
    """b + (scale + zero point bits) / G."""
    return bits + meta_bits / group


def kivi_qdq(k: torch.Tensor, v: torch.Tensor, bits: int = 4, group: int = 32, residual: int = 128,
             key_axis: str = "channel", values: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """k, v (B, KV, S, d): quantise the complete G-token groups that lie entirely before the last ``residual``
    tokens; keep the rest exact. Keys: per channel (groups along the token axis) or per token; values per token
    (groups of ``group`` channels along d, or all of d if d < group)."""
    S = k.shape[2]
    n_q = max(0, (S - residual) // group) * group              # tokens in complete groups outside the window
    if n_q == 0:
        return k, v
    k_old, v_old = k[:, :, :n_q], v[:, :, :n_q]
    d = k.shape[-1]
    gd = group if d % group == 0 else d
    kq = asym_qdq(k_old, bits, 2, group) if key_axis == "channel" else asym_qdq(k_old, bits, 3, gd)
    vq = asym_qdq(v_old, bits, 3, gd) if values else v_old
    return torch.cat([kq, k[:, :, n_q:]], 2), torch.cat([vq, v[:, :, n_q:]], 2)


def kv_cache_bits(S: int, elements_per_token: int, bits: int, group: int, residual: int,
                  full_bits: int = 16) -> float:
    """Total cache bits of one sequence and layer under KIVI's layout (S tokens, residual exact)."""
    n_q = max(0, (S - residual) // group) * group
    return n_q * elements_per_token * bits_per_element(bits, group) + (S - n_q) * elements_per_token * full_bits


# --------------------------------------------------------------------------------------------- attention kind

@register("gqa-kvq")
class QuantKVAttention(GQAttention):
    """GQA whose cached keys and values are seen through ``kivi_qdq`` (module docstring)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        ex = cfg.extra
        self.kv_bits = int(ex.get("kv_bits", 4))
        self.kv_group = int(ex.get("kv_group", 32))
        self.kv_residual = int(ex.get("kv_residual", 128))
        self.kv_key_axis = ex.get("kv_key_axis", "channel")
        self.kv_values = bool(ex.get("kv_quant_values", True))

    def forward(self, x: torch.Tensor, positions: torch.Tensor, cache=None):
        if cache is None:
            return super().forward(x, positions, None)
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.H, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.KV, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.KV, self.hd).transpose(1, 2)
        cos, sin = self.rope(positions)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k_pos = positions
        if "k" in cache:
            k = torch.cat((cache["k"], k), dim=2)
            v = torch.cat((cache["v"], v), dim=2)
            k_pos = torch.cat((cache["pos"], positions))
        cache["k"], cache["v"], cache["pos"] = k, v, k_pos
        kq, vq = kivi_qdq(k, v, self.kv_bits, self.kv_group, self.kv_residual, self.kv_key_axis, self.kv_values)
        mask = causal_mask(positions, k_pos)
        y = F.scaled_dot_product_attention(q, kq, vq, attn_mask=mask, enable_gqa=self.H != self.KV)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.hd))


def plain_copy(model, attention: str = "gqa", **extra):
    """A plain ``LM`` with the same main-path weights (drops MTP modules or other Module 6 extras)."""
    import copy
    from frontierlab.model import LM
    ex = {k: v for k, v in model.config.extra.items() if k != "blocks"}
    m = LM(model.config.with_(attention=attention, extra={**ex, **extra}))
    sd = {k: v for k, v in copy.deepcopy(model.state_dict()).items() if k in m.state_dict()}
    m.load_state_dict(sd)
    return m.eval()


def with_quant_kv(model, bits: int = 4, group: int = 32, residual: int = 128, key_axis: str = "channel",
                  values: bool = True):
    """A copy of a GQA model whose attention modules use the quantised cache (same weights)."""
    return plain_copy(model, "gqa-kvq", kv_bits=bits, kv_group=group, kv_residual=residual, kv_key_axis=key_axis,
                      kv_quant_values=values)


@torch.no_grad()
def decode_nll(model, windows: torch.Tensor, prefill: int = 0, ref=None) -> dict:
    """Teacher-forced decoding of ``windows`` (B, T): prefill ``prefill`` tokens in one forward, then one token
    per step with the cache. Returns per-window mean NLL of the tokens after the prefill and, if ``ref`` is
    given (a model decoded the same way), the mean KL(ref || model) of the next-token distributions."""
    model.eval()
    B, T = windows.shape
    cache = model.new_cache()
    rcache = ref.new_cache() if ref is not None else None
    nll = torch.zeros(B, dtype=torch.float64)
    kl = torch.zeros(B, dtype=torch.float64)
    start = max(1, prefill)
    lg = model(windows[:, :start], cache=cache).logits[:, -1].double()
    rl = ref(windows[:, :start], cache=rcache).logits[:, -1].double() if ref is not None else None
    for t in range(start, T):
        lp = torch.log_softmax(lg, -1)
        nll -= lp.gather(-1, windows[:, t:t + 1]).squeeze(-1)
        if ref is not None:
            rp = torch.log_softmax(rl, -1)
            kl += (rp.exp() * (rp - lp)).sum(-1)
        if t < T - 1:
            lg = model(windows[:, t:t + 1], cache=cache).logits[:, -1].double()
            if ref is not None:
                rl = ref(windows[:, t:t + 1], cache=rcache).logits[:, -1].double()
    n = T - start
    return {"nll": (nll / n).tolist(), "kl": (kl / n).tolist() if ref is not None else None}


# --------------------------------------------------------------------------------------------- weights

def block_linears(model: nn.Module) -> list[str]:
    """Every nn.Linear inside a transformer block (not the embedding, not the head)."""
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and ".layers." in f".{n}"
            and not n.endswith("lm_head")]


@torch.no_grad()
def quantize_weights_(model: nn.Module, bits: int = 4, group: int = 128, symmetric: bool = False) -> dict:
    """Round-to-nearest weight quantisation in place, groups of ``group`` input channels per output row.

    Asymmetric uses :func:`asym_qdq`; symmetric uses Module 8's INT grid (``precision.quant``). Returns the
    number of matrices, mean relative error and storage bits per weight (scales and zero points counted)."""
    from frontierlab.precision.quant import QuantSpec, qdq
    errs, n = [], 0
    for name in block_linears(model):
        m = model.get_submodule(name)
        W = m.weight
        g = group if W.shape[1] % group == 0 else W.shape[1]
        if symmetric:
            Wq = qdq(W, QuantSpec(f"int{bits}", (1, g), "fp32"))
        else:
            Wq = asym_qdq(W, bits, 1, g)
        errs.append(float((Wq - W).norm() / W.norm()))
        n += W.numel()
        W.copy_(Wq)
    meta = 32                       # fp32 scale (symmetric) or fp16 scale + fp16 zero point (asymmetric)
    return {"matrices": len(errs), "mean_rel_err": sum(errs) / max(1, len(errs)), "params": n,
            "bits_per_weight": bits + meta / group}
