"""Real FP8 training kernels through torchao (main path only; lesson 08.2).

torchao 0.18.0 (pinned in references/versions.md; API checked at tag v0.18.0, file
``torchao/float8/float8_linear_utils.py``):

    from torchao.float8 import convert_to_float8_training, Float8LinearConfig
    config = Float8LinearConfig.from_recipe_name("tensorwise")      # or "rowwise", "rowwise_with_gw_hp"
    convert_to_float8_training(model, module_filter_fn=fn, config=config)   # fn(module, fqn) -> bool

It swaps the selected ``nn.Linear`` modules for ``Float8Linear`` (same parameter names), whose three GEMMs call
``torch._scaled_mm`` on FP8 operands. The GEMM needs both dimensions of each weight to be multiples of 16; the
filter below skips any that are not. Use ``torch.compile`` (``--compile`` in the loop): torchao's docs pair the
recipes with compile "for competitive performance", because the casts and amax reductions are separate kernels
otherwise. FP8 tensor cores exist on Ada (sm89, e.g. L4) and Hopper (sm90, H100) and later; torchao's own tests
gate on sm89+ (INFERENCE that it will not run below that; it is not stated in the docs).

Nothing here runs on a CPU, and nothing here was run in this build: it is part of the Module 8 pilot.
"""

from __future__ import annotations

import torch

from frontierlab.precision.linear import select_linears

TORCHAO_RECIPES = ("tensorwise", "rowwise", "rowwise_with_gw_hp")


def fp8_capable(device=None) -> bool:
    """True on a CUDA device with compute capability >= 8.9 (FP8 tensor cores)."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return (major, minor) >= (8, 9)


def convert_torchao_float8(model, recipe: str = "tensorwise", keep_blocks=frozenset(), only: str = "all") -> list[str]:
    """Swap the same linears the emulated recipes would quantise for torchao ``Float8Linear``. CUDA only."""
    if recipe not in TORCHAO_RECIPES:
        raise ValueError(f"torchao recipe must be one of {TORCHAO_RECIPES}")
    if not torch.cuda.is_available():
        raise RuntimeError("torchao Float8 training needs a CUDA GPU with FP8 tensor cores (L4, H100, ...); "
                           "on a CPU use an emulated recipe (--recipe fp8-tensorwise / fp8-rowwise), labelled emulation")
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    wanted = set(select_linears(model, keep_blocks, only))
    chosen: list[str] = []

    def keep(mod, fqn: str) -> bool:
        ok = fqn in wanted and mod.in_features % 16 == 0 and mod.out_features % 16 == 0
        if ok:
            chosen.append(fqn)
        return ok

    convert_to_float8_training(model, module_filter_fn=keep, config=Float8LinearConfig.from_recipe_name(recipe))
    return sorted(set(chosen))
