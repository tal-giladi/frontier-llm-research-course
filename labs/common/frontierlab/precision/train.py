"""Module 8 training runs: the course loop with emulated (or, on CUDA, real) low-precision linear layers.

    python -m frontierlab.precision.train --recipe fp8-deepseek --run runs/m08/fp8 --preset toy --steps 300
    python -m frontierlab.precision.train --recipe nvfp4 --keep-high last1 --precision-log --run ... --preset toy
    python -m frontierlab.precision.train --torchao rowwise --compile --dtype bf16 --run ... --preset pilot-30m   # GPU

It calls the Module 7 wrapper (``frontierlab.optim.train``), so every loop argument and every Module 7 argument
(``--optimizer muon``, ``--stability-log``, ``--width``, ``--init-from`` ...) still works; with the defaults that
wrapper runs the loop's own AdamW unchanged. For the duration of one call this wrapper replaces
``frontierlab.optim.train.build_model`` (to swap linears after the model is built, so the initial weights are the
same draws as an unswapped run) and ``loop.write_run_card`` (to add a ``precision`` block), and runs ``loop.param_counts`` / ``loop.flops_per_token``
with the torch RNG forked (they build modules after the checkpoint's RNG state is restored, which would break
exact resume under stochastic rounding; the loop fix is in the inbox). Neither
``train/loop.py`` nor the model files are edited; the native flags are proposed in
``curriculum/inbox/module-08-shared-changes.md``.

Options:

* ``--recipe NAME`` — one of :data:`frontierlab.precision.linear.RECIPES` (default ``bf16``: nothing swapped).
  Every recipe except ``bf16`` is an **emulation**: it reproduces the recipe's rounding in float32 and is slower
  than the baseline, never faster. The run card says ``emulated: true``.
* ``--keep-high "first2,last8"`` — transformer blocks left in high precision (the NVFP4 recipe's choice for its
  12B model); ``--only {all,mlp,attn}`` — which linears of the other blocks are quantised. The embedding and
  output head are never quantised.
* ``--torchao {tensorwise,rowwise,rowwise_with_gw_hp}`` — real FP8 GEMMs from torchao 0.18.0 on a CUDA GPU with
  FP8 tensor cores (L4, H100). Not emulation; the only path here whose speed means anything.
* ``--precision-log [--precision-every N]`` — ``<run>/precision.jsonl`` (``frontierlab.precision.monitor``).

Exact resume holds (``labs/common/tests/test_precision.py``): the swapped layers keep the parameter names, and
stochastic rounding draws from the global torch RNG, which every checkpoint stores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from frontierlab.optim import train as optim_train
from frontierlab.precision.linear import RECIPES, get_recipe, parse_keep, swap_linears
from frontierlab.precision.monitor import PrecisionLogger
from frontierlab.precision.torchao_path import TORCHAO_RECIPES, convert_torchao_float8
from frontierlab.train import loop


def build_parser():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--recipe", default="bf16", help=f"one of {sorted(RECIPES)} or w-int2 .. w-int8")
    ap.add_argument("--keep-high", default="", help='blocks kept in high precision, e.g. "first2,last8"')
    ap.add_argument("--only", choices=["all", "mlp", "attn"], default="all")
    ap.add_argument("--torchao", choices=TORCHAO_RECIPES, default=None)
    ap.add_argument("--precision-log", action="store_true")
    ap.add_argument("--precision-every", type=int, default=10)
    return ap


def apply_precision(model, recipe: str = "bf16", keep_high: str = "", only: str = "all", torchao: str | None = None) -> dict:
    """Swap a built model's linears in place. Returns the run card's ``precision`` block."""
    L = model.config.num_hidden_layers
    keep = parse_keep(keep_high, L)
    if torchao:
        names = convert_torchao_float8(model, torchao, keep, only)
    else:
        names = swap_linears(model, recipe, keep, only)
    r = get_recipe(recipe)
    return {"recipe": recipe, "torchao": torchao, "emulated": bool(r.emulated and not torchao),
            "keep_high": keep_high, "kept_blocks": sorted(keep), "only": only, "quantized_linears": len(names),
            "specs": r.specs(), "rht": r.rht, "notes": r.notes}


def load_model(checkpoint, recipe: str | None = None, keep_high: str | None = None, only: str | None = None,
               map_location="cpu", torchao: str | None = None):
    """Rebuild a Module 8 model from a checkpoint with the same emulated layers (read from the run card next to it
    unless given). Evaluate an emulated-precision run with the same precision it trained with; a torchao run is
    converted again (on CUDA). Pass ``recipe="bf16"`` to evaluate the master weights in high precision."""
    from frontierlab.runcard import read_run_card
    model = optim_train.load_model(checkpoint, map_location)
    card_dir = Path(checkpoint).parent
    prec = {}
    if (card_dir / "run_card.yaml").exists():
        prec = read_run_card(card_dir).get("precision") or {}
    if recipe is None and torchao is None:
        torchao = prec.get("torchao")
    if torchao:
        model.to("cuda")
    apply_precision(model, recipe or prec.get("recipe", "bf16"), keep_high if keep_high is not None else prec.get("keep_high", ""),
                    only or prec.get("only", "all"), torchao)
    return model


def main(argv=None):
    mine, rest = build_parser().parse_known_args(argv)
    get_recipe(mine.recipe)                                            # fail early on an unknown recipe
    if any(x in ("-h", "--help") for x in rest):
        print(__doc__)
        return None
    _, loop_rest = optim_train.build_parser().parse_known_args(rest)
    a0 = loop.build_parser().parse_args(loop_rest)                     # fail early on a typo
    if mine.torchao and mine.recipe != "bf16":
        raise SystemExit("--torchao replaces the linears with real FP8 ones; do not combine it with an emulated --recipe")
    if mine.torchao and not a0.device.startswith("cuda"):
        raise SystemExit("--torchao needs a CUDA GPU with FP8 tensor cores; on a CPU use --recipe fp8-tensorwise "
                         "or fp8-rowwise (emulation)")
    if mine.precision_log and a0.compile:
        raise SystemExit("--precision-log uses forward hooks and is not supported with --compile")
    holder: dict = {}

    def build_model(cfg):
        model = saved_build(cfg)
        holder["info"] = apply_precision(model, mine.recipe, mine.keep_high, mine.only, mine.torchao)
        if mine.precision_log:
            holder["log"] = PrecisionLogger(model, a0.run / "precision.jsonl", mine.precision_every)
        return model

    def write_card(run_dir, **kw):
        kw["extra"] = {**(kw.get("extra") or {}), "precision": holder.get("info", {})}
        return saved["write_run_card"](run_dir, **kw)

    def rng_neutral(fn):
        """``accounting.param_counts`` instantiates attention modules (random init) AFTER the loop restored the
        torch RNG from the checkpoint. Harmless when training draws no random numbers; with stochastic rounding
        it would make a resumed run differ from a straight one. Run the accounting with the RNG forked."""
        def wrapped(*args, **kw):
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                return fn(*args, **kw)
        return wrapped

    names = ("param_counts", "flops_per_token", "write_run_card")
    saved = {n: getattr(loop, n) for n in names}
    saved_build = optim_train.build_model
    optim_train.build_model = build_model
    loop.param_counts, loop.flops_per_token = rng_neutral(saved["param_counts"]), rng_neutral(saved["flops_per_token"])
    loop.write_run_card = write_card
    try:
        return optim_train.main(rest)
    finally:
        optim_train.build_model = saved_build
        for n, f in saved.items():
            setattr(loop, n, f)
        if "log" in holder:
            holder["log"].close()


if __name__ == "__main__":
    main()
