"""Train one Module 3 arm with the course training loop (``frontierlab.train.loop``), unchanged.

    python labs/module-03/train_variant.py --arm mla -- --run runs/m03/mla-s0 --preset toy --steps 400 \\
        --batch 16 --seq 128
    python labs/module-03/train_variant.py --arm mla --match-params -- --run ... (same loop arguments)
    python labs/module-03/train_variant.py --arm mla --equal-flops-to 400 -- --run ... --steps 400 ...

Everything after ``--`` goes to the loop as it is. This wrapper only:

1. builds the arm's config from the preset (``m03.arm_config``), optionally with the SwiGLU width
   changed so non-embedding parameters equal the preset's (``--match-params``, equal-parameters axis);
2. with ``--equal-flops-to N``, replaces ``--steps`` by the step count whose training FLOPs equal the
   preset's at N steps (same tokens per step; equal-FLOPs axis);
3. makes the loop's run card and MFU use ``frontierlab.attention.accounting`` (exact for every kind)
   instead of the GQA-only formulas, so ``budget.train_flops`` and ``params`` are right for MLA.

The run card records the whole config (``config.attention``, ``config.extra``,
``config.intermediate_size``), so ``python -m frontierlab.record`` sees exactly what changed.
Exact resume works as in lesson 01.1: rerun the same command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m03  # noqa: E402
from frontierlab.attention import accounting  # noqa: E402
from frontierlab.model import LM  # noqa: E402
from frontierlab.train import loop  # noqa: E402


def plan(arm: str, loop_argv: list[str], match_params: bool = False, equal_flops_to: int | None = None):
    """(config, loop argv) for one arm. Reads the vocabulary from Data-v0's meta.json (or --data)."""
    a = loop.build_parser().parse_args(loop_argv)
    from frontierlab.data.loader import TokenData
    vocab = TokenData("train", **({"root": a.data} if a.data else {})).meta["vocab_size"]
    base = m03.preset_config(a.preset, vocab)
    cfg = m03.arm_config(arm, base, a.seq, match_params=match_params)
    argv = list(loop_argv)
    if equal_flops_to is not None:
        steps = m03.equal_flops_steps(cfg, base, equal_flops_to, a.seq)
        if "--steps" in argv:
            argv[argv.index("--steps") + 1] = str(steps)
        else:
            argv += ["--steps", str(steps)]
    return cfg, argv


def train(cfg, argv):
    """Run the unmodified loop with ``cfg`` as the model config and exact accounting for its card."""
    loop.make_model = lambda a, vocab_size: (cfg, LM(cfg))
    loop.param_counts = lambda c: {k: v for k, v in accounting.param_counts(c).items()
                                   if k in ("total", "non_embedding", "embedding")}
    loop.flops_per_token = accounting.flops_per_token
    return loop.main(argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        raise SystemExit("usage: train_variant.py --arm ARM [--match-params] [--equal-flops-to N] -- <loop args>")
    i = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=m03.ARMS, required=True)
    ap.add_argument("--match-params", action="store_true")
    ap.add_argument("--equal-flops-to", type=int, default=None)
    w = ap.parse_args(argv[:i])
    cfg, loop_argv = plan(w.arm, argv[i + 1:], w.match_params, w.equal_flops_to)
    a = loop.build_parser().parse_args(loop_argv)
    print(f"arm {w.arm}: {m03.describe(cfg, a.seq)}  steps {a.steps}")
    return train(cfg, loop_argv)


if __name__ == "__main__":
    main()
