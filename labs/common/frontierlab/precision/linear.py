"""An emulated low-precision linear layer whose three GEMMs follow a training recipe (lessons 08.2, 08.3).

Training one linear layer y = x Wᵀ runs three GEMMs (x: (M, K) tokens × in-features, W: (N, K)):

    Fprop   y  = x  Wᵀ      contracts over K (in-features)
    Dgrad   dx = dy W       contracts over N (out-features)
    Wgrad   dW = dyᵀ x      contracts over M (tokens)

A block-scaled GEMM needs each operand's scales along *its contraction dimension*, so the same tensor is
quantised differently for different GEMMs: x along K for Fprop but along M for Wgrad (DeepSeek-V3 appendix B.2:
1 × 128 tiles in the forward pass, 128 × 1 in the backward pass); W along K for Fprop but along N for Dgrad. With
1-D weight blocks the Dgrad weight is therefore a *different* tensor from the Fprop weight — the backward pass
differentiates a function the forward pass did not compute. 2-D 16 × 16 weight blocks (NVFP4) or 128 × 128
blocks (DeepSeek-V3) quantise W and Wᵀ identically, which is why those recipes use them.

:class:`QLinear` is an ``nn.Linear`` (same parameter names, so checkpoints and exact resume are unchanged) whose
forward calls :class:`_QLinearFn`. Gradients pass through every quantiser unchanged (straight-through): the
backward computes dx and dW from the quantised operands, exactly as the recipes' kernels do, and hands them to
the high-precision master weights. Each GEMM is computed in float32 on dequantised values — the products of
FP8/FP4 values are exact in float32 and the sum is an FP32 accumulation (DeepSeek-V3's promotion target), so this
is an emulation of the recipe's numerics with exact FP32 accumulation. It measures numerics, never speed.

Recipes (:data:`RECIPES`):

    fp8-tensorwise   E4M3 x and W, E5M2 dy, one fp32 scale per tensor       torchao "tensorwise" numerics
    fp8-rowwise      E4M3 everywhere, per-row power-of-2 scales along each GEMM's contraction dim   torchao "rowwise"
    fp8-deepseek     E4M3 everywhere; x and dy in 1 × 128 tiles, W in 128 × 128 blocks, fp32 scales   DeepSeek-V3 §3.3
    mxfp8            E4M3, 1 × 32 blocks, E8M0 scales (OCP MX)
    mxfp4            E2M1, 1 × 32 blocks, E8M0 scales, RNE everywhere (no SR, no RHT): a naive FP4 baseline
    nvfp4            E2M1; x and dy 1 × 16, W 16 × 16, two-level scales; RHT (16) on Wgrad inputs; SR on dy
                     (arXiv 2509.25149 section 4)
    nvfp4-no-rht, nvfp4-no-sr, nvfp4-1d-weights    the recipe with one ingredient removed (ablations)
    int4-qat, mxfp4-qat    weight-only fake quantisation for quantisation-aware training (forward weight
                     quantised, everything else high precision; dx uses the quantised weight, dW the plain STE)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from frontierlab.precision.hadamard import apply_rht
from frontierlab.precision.quant import QuantSpec, get_spec, qdq, quant_error


@dataclass(frozen=True)
class Recipe:
    name: str
    act: QuantSpec | None = None          # x for Fprop (blocks along K)
    weight: QuantSpec | None = None       # W for Fprop (blocks along K) and, re-quantised along N, for Dgrad
    grad: QuantSpec | None = None         # dy for Dgrad (blocks along N)
    wgrad_grad: QuantSpec | None = None   # dyᵀ for Wgrad (blocks along M); default: grad
    wgrad_act: QuantSpec | None = None    # xᵀ for Wgrad (blocks along M); default: act
    dgrad_weight: str = "requant"         # "requant": quantise Wᵀ along N; "forward": reuse the Fprop weight (QAT)
    rht: int = 0                          # Hadamard size on Wgrad inputs (0 = off)
    rht_seed: int = 0                     # one sign vector shared by every layer (NVFP4 recipe)
    notes: str = ""

    def with_(self, **kw) -> "Recipe":
        return replace(self, **kw)

    def specs(self) -> dict:
        return {k: (None if getattr(self, k) is None else vars(getattr(self, k)).copy())
                for k in ("act", "weight", "grad", "wgrad_grad", "wgrad_act")}

    @property
    def emulated(self) -> bool:
        return any(getattr(self, k) is not None for k in ("act", "weight", "grad", "wgrad_grad", "wgrad_act"))


def _s(name, **kw) -> QuantSpec:
    return get_spec(name).with_(**kw) if kw else get_spec(name)


E4 = QuantSpec("e4m3", (0, 0), "fp32")
E5 = QuantSpec("e5m2", (0, 0), "fp32")
ROW = QuantSpec("e4m3", (1, 0), "pow2")
TILE = _s("fp8-tile128")
BLK = _s("fp8-block128")
MX8 = _s("mxfp8")
MX4 = _s("mxfp4")
NV1 = _s("nvfp4-1d")
NV2 = _s("nvfp4-2d")
NV1_SR = NV1.with_(rounding="sr")

RECIPES: dict[str, Recipe] = {r.name: r for r in (
    Recipe("bf16", notes="no quantisation: the baseline (plain nn.Linear, not swapped)"),
    Recipe("fp8-tensorwise", E4, E4, E5, notes="torchao tensorwise numerics: E4M3 forward, E5M2 gradients"),
    Recipe("fp8-rowwise", ROW, ROW, ROW, notes="torchao rowwise numerics: E4M3, per-row power-of-2 scales"),
    Recipe("fp8-deepseek", TILE, BLK, TILE, notes="DeepSeek-V3 section 3.3.2: 1x128 tiles, 128x128 weight blocks"),
    Recipe("mxfp8", MX8, MX8, MX8, notes="OCP MX FP8 (E4M3), 1x32 blocks, E8M0 scales"),
    Recipe("mxfp4", MX4, MX4, MX4, notes="OCP MXFP4, 1x32 blocks, E8M0 scales, RNE, no RHT: naive FP4"),
    Recipe("nvfp4", NV1, NV2, NV1_SR, rht=16, notes="arXiv 2509.25149 section 4"),
    Recipe("nvfp4-no-rht", NV1, NV2, NV1_SR, rht=0, notes="ablation: no random Hadamard transform"),
    Recipe("nvfp4-no-sr", NV1, NV2, NV1, rht=16, notes="ablation: round-to-nearest gradients"),
    Recipe("nvfp4-1d-weights", NV1, NV1, NV1_SR, rht=16, notes="ablation: 1x16 weight blocks (W and W^T differ)"),
    Recipe("int4-qat", None, _s("int4-g32"), None, dgrad_weight="forward",
           notes="weight-only INT4, symmetric, group 32 (the Kimi-K2-Thinking config.json scheme), STE"),
    Recipe("mxfp4-qat", None, MX4, None, dgrad_weight="forward", notes="weight-only MXFP4 fake quantisation, STE"),
)}


def get_recipe(r) -> Recipe:
    """A recipe by name. Besides :data:`RECIPES`, ``"w-int<b>"`` (b = 2..8) is weight-only INT-b fake quantisation
    with one fp32 scale per output row, for the precision-scaling sweep of lesson 08.4 (the integer-type,
    weight-trained-in-low-precision setting of arXiv 2411.04330 Eq. 3)."""
    if isinstance(r, Recipe):
        return r
    if isinstance(r, str) and r.startswith("w-int") and r[5:].isdigit() and 2 <= int(r[5:]) <= 8:
        return Recipe(r, None, QuantSpec(f"int{r[5:]}", (1, 0), "fp32"), None, dgrad_weight="forward",
                      notes=f"weight-only INT{r[5:]} per-row fake quantisation (QAT), STE")
    if r not in RECIPES:
        raise KeyError(f"unknown recipe {r!r}; known: {sorted(RECIPES)}")
    return RECIPES[r]


class _QLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2, w, recipe: Recipe, stats: dict | None):
        with torch.autocast(device_type=x2.device.type, enabled=False):   # emulate in fp32, never bf16
            return _QLinearFn._fwd(ctx, x2, w, recipe, stats)

    @staticmethod
    def _fwd(ctx, x2, w, recipe, stats):
        xq = qdq(x2, recipe.act)
        wq = qdq(w, recipe.weight)
        wd = torch.float64 if x2.dtype == torch.float64 else torch.float32       # fp32 accumulation (fp64 in tests)
        y = xq.to(wd) @ wq.to(wd).t()
        if stats is not None:
            if recipe.act is not None:
                stats["act"] = quant_error(x2, recipe.act)
            if recipe.weight is not None:
                stats["weight"] = quant_error(w, recipe.weight)
        ctx.recipe, ctx.stats = recipe, stats
        ctx.save_for_backward(x2, w, wq)
        return y.to(x2.dtype)

    @staticmethod
    def backward(ctx, gy):
        with torch.autocast(device_type=gy.device.type, enabled=False):
            return _QLinearFn._bwd(ctx, gy)

    @staticmethod
    def _bwd(ctx, gy):
        """Stochastic rounding (dy in the NVFP4 recipe) draws from the global torch RNG, which the loop checkpoints."""
        x2, w, wq = ctx.saved_tensors
        r: Recipe = ctx.recipe
        dx = dw = None
        if ctx.needs_input_grad[0]:
            gq = qdq(gy, r.grad)                                             # dy along N
            if r.dgrad_weight == "forward" or r.weight is None:
                wd = wq
            else:
                wd = qdq(w.t(), r.weight).t()                                # Wᵀ (K, N) quantised along N
            ct = torch.float64 if gy.dtype == torch.float64 else torch.float32
            dx = (gq.to(ct) @ wd.to(ct)).to(x2.dtype)
        if ctx.needs_input_grad[1]:
            gt, xt = gy.t(), x2.t()                                          # (N, M), (K, M): contraction over M
            gws = r.wgrad_grad if r.wgrad_grad is not None else r.grad
            xws = r.wgrad_act if r.wgrad_act is not None else r.act
            if r.rht and (gws is not None or xws is not None):
                gt, xt = apply_rht(gt, r.rht, r.rht_seed), apply_rht(xt, r.rht, r.rht_seed)
            gtq, xtq = qdq(gt, gws), qdq(xt, xws)
            ct = torch.float64 if gy.dtype == torch.float64 else torch.float32
            dw = (gtq.to(ct) @ xtq.to(ct).t()).to(w.dtype)
            if ctx.stats is not None and gws is not None:
                ctx.stats["grad"] = quant_error(gt, gws)
        return dx, dw, None, None


class QLinear(nn.Linear):
    """``nn.Linear`` (no bias) running its GEMMs through a :class:`Recipe`. Build with :meth:`from_linear`."""

    recipe: Recipe

    @classmethod
    def from_linear(cls, lin: nn.Linear, recipe) -> "QLinear":
        if lin.bias is not None:
            raise ValueError("QLinear emulates bias-free linears (every frontierlab linear is bias-free)")
        q = cls.__new__(cls)
        nn.Module.__init__(q)
        q.in_features, q.out_features = lin.in_features, lin.out_features
        q.weight = lin.weight                                  # the same Parameter object: same name, same storage
        q.register_parameter("bias", None)
        q.recipe = get_recipe(recipe)
        q.collect = False
        q.last_stats = {}
        return q

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        stats = {} if self.collect else None
        y = _QLinearFn.apply(x2, self.weight, self.recipe, stats)
        if stats is not None:
            self.last_stats = stats
        return y.reshape(*shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, recipe={self.recipe.name}"


# --------------------------------------------------------------------------------------------- swapping

def block_index(name: str) -> int | None:
    parts = name.split(".")
    if "layers" in parts:
        i = parts.index("layers")
        if i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def parse_keep(spec: str | None, num_layers: int) -> set[int]:
    """Blocks kept in high precision: "first2,last8" -> {0, 1, L-8, ..., L-1}; "" -> none; "3,5" -> {3, 5}."""
    keep: set[int] = set()
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("first"):
            keep |= set(range(min(num_layers, int(part[5:]))))
        elif part.startswith("last"):
            keep |= set(range(max(0, num_layers - int(part[4:])), num_layers))
        else:
            keep.add(int(part))
    return keep


def select_linears(model: nn.Module, keep_blocks: set[int] = frozenset(), only: str = "all") -> list[str]:
    """Names of the linears a recipe applies to: every ``nn.Linear`` inside a transformer block, except blocks in
    ``keep_blocks``; never the embedding or the output head (DeepSeek-V3 section 3.3.1 keeps both, and the
    course's head is tied to the embedding). ``only``: "all", "mlp" (SwiGLU matrices: the analogue of MoE expert
    weights in a dense model) or "attn" (q, k, v, o projections)."""
    out = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear) or name.endswith("lm_head"):
            continue
        b = block_index(name)
        if b is None or b in keep_blocks:
            continue
        if only == "mlp" and ".mlp." not in f".{name}.":
            continue
        if only == "attn" and ".self_attn." not in f".{name}.":
            continue
        out.append(name)
    return out


def swap_linears(model: nn.Module, recipe, keep_blocks: set[int] = frozenset(), only: str = "all") -> list[str]:
    """Replace the selected ``nn.Linear`` modules by :class:`QLinear` with ``recipe``, in place, keeping the same
    weight Parameters (state-dict keys and values unchanged). Returns the swapped names. ``recipe="bf16"`` swaps nothing."""
    r = get_recipe(recipe)
    if not r.emulated:
        return []
    names = select_linears(model, keep_blocks, only)
    for name in names:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, QLinear.from_linear(getattr(parent, child), r))
    return names


@torch.no_grad()
def ptq_(model: nn.Module, spec, keep_blocks: set[int] = frozenset(), only: str = "all") -> dict:
    """Post-training quantisation, in place: every selected linear's weight becomes qdq(W) (round to nearest).
    Returns the mean relative weight error and the storage bits per weight of the quantised matrices."""
    spec = get_spec(spec)
    errs, n_q, bits = [], 0, 0.0
    for name in select_linears(model, keep_blocks, only):
        m = model.get_submodule(name)
        e = quant_error(m.weight, spec)
        errs.append(e["rel_err"])
        m.weight.copy_(qdq(m.weight, spec))
        n_q += m.weight.numel()
        bits += e["bits"] * m.weight.numel()
    return {"matrices": len(errs), "mean_rel_err": sum(errs) / max(1, len(errs)), "params": n_q,
            "bits_per_weight": bits / max(1, n_q)}
