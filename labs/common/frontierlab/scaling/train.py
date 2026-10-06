"""Module 11 training runs: any course wrapper, with the ladder sizes registered and an optional unique-data cap.

    python -m frontierlab.scaling.train --run runs/m11/r3-c1e12 --preset m11-r3 --steps 120 --batch 8 --seq 128
    python -m frontierlab.scaling.train --unique-tokens 65536 --run runs/m11/rep16 --preset m11-r3 --steps 600 ...
    python -m frontierlab.scaling.train --via optim --optimizer muon --run ... --preset m11-r4 ...     # Recipe-R
    python -m frontierlab.scaling.train --via datax --mixture data_v1.json --run ... --preset m11-r4 ...

``--via loop`` (default) runs ``frontierlab.train.loop``; ``--via optim`` runs ``frontierlab.optim.train``
(Module 7: Muon, µP, schedules); ``--via datax`` runs ``frontierlab.datax.train`` (Module 10: mixtures,
document masking, anneals). Every other argument goes to that module unchanged, so exact resume, run cards and
``metrics.jsonl`` are the loop's own. Importing :mod:`frontierlab.scaling.ladder` first adds the ladder presets
(``m11-r1`` … ``m11-r6`` on CPU, ``m11-200m``, ``m11-350m``, ``m11-1b`` on the main path).

``--unique-tokens U`` (``--via loop`` only): training windows are drawn only from the first U tokens of the
training split, so a run of D tokens sees its data about D / U times: the repetition experiment of lesson
11.1 (Muennighoff et al., arXiv 2305.16264). The generator is used exactly as the loop uses it (one
``randint`` per batch), so resume stays exact. The run card gets ``scaling: {unique_tokens, epochs}``.
"""

from __future__ import annotations

import argparse

import torch

from frontierlab.data.loader import TokenData
from frontierlab.scaling import ladder  # noqa: F401  (registers the presets)
from frontierlab.train import loop


def build_parser():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--via", choices=["loop", "optim", "datax"], default="loop")
    ap.add_argument("--unique-tokens", type=int, default=None)
    return ap


class CappedTokenData(TokenData):
    """Training windows from the first ``cap`` tokens only (validation and test are untouched)."""

    cap: int | None = None

    def batch(self, B, T, generator, device="cpu"):
        if self.split != "train" or self.cap is None:
            return super().batch(B, T, generator, device)
        hi = min(self.cap, len(self.tokens)) - T
        starts = torch.randint(0, hi, (B,), generator=generator).tolist()
        return torch.stack([self.window(s, T) for s in starts]).to(device)


def main(argv=None):
    mine, rest = build_parser().parse_known_args(argv)
    if mine.via == "optim":
        if mine.unique_tokens:
            raise SystemExit("--unique-tokens is implemented for --via loop only")
        from frontierlab.optim import train as otrain
        return otrain.main(rest)
    if mine.via == "datax":
        if mine.unique_tokens:
            raise SystemExit("--unique-tokens is implemented for --via loop only")
        from frontierlab.datax import train as dtrain
        return dtrain.main(rest)
    if mine.unique_tokens is None:
        return loop.main(rest)
    a0 = loop.build_parser().parse_args(rest)
    CappedTokenData.cap = mine.unique_tokens
    epochs = a0.steps * a0.batch * a0.seq * a0.grad_accum / mine.unique_tokens

    def write_card(run_dir, **kw):
        kw["extra"] = {**(kw.get("extra") or {}), "scaling": {"unique_tokens": mine.unique_tokens, "epochs": epochs}}
        return saved[1](run_dir, **kw)

    saved = (loop.TokenData, loop.write_run_card)
    loop.TokenData, loop.write_run_card = CappedTokenData, write_card
    try:
        return loop.main(rest)
    finally:
        loop.TokenData, loop.write_run_card = saved
        CappedTokenData.cap = None


if __name__ == "__main__":
    main()
