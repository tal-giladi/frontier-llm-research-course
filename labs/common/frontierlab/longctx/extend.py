"""Continued training at a longer context from a checkpoint (lesson 04.3): a wrapper around the course loop.

    python -m frontierlab.longctx.extend --init-from runs/m04/base256/checkpoint.pt \\
        --rope yarn --factor 4 --original 256 --long-fraction 1.0 \\
        --run runs/m04/ext-yarn --preset toy --steps 200 --batch 4 --seq 1024 --lr 1e-3 --warmup 20

Every argument the loop knows (``--run``, ``--steps``, ``--seq``, ``--lr``, ``--max-minutes``, ...) is
passed through to ``frontierlab.train.loop`` unchanged. This module adds what continued training
needs and the loop does not have yet (the exact loop change is proposed in
``curriculum/inbox/module-04-loop-changes.md``):

* ``--init-from CKPT``: start from the weights of a checkpoint written by the loop. Only the model
  weights are taken. The optimizer starts fresh (new AdamW moments, warmup from step 0): the usual
  choice when the data distribution and sequence length change, and the one the course uses.
  The model configuration comes from the checkpoint and must match ``--preset``.
* the RoPE rule of the extended model: ``--rope {default,pi,ntk,yarn,llama3}``, ``--factor``,
  ``--original`` (the trained length), ``--beta-fast``, ``--beta-slow``, ``--ramp``,
  ``--attention-factor``; or ``--theta`` to change the RoPE base instead ("base-frequency
  adjustment", the Llama 3 / Code Llama style of extension); ``--nope-every N`` switches to the
  ``"gqa-irope"`` kind (only from a model *trained* with that layout).
* ``--long-fraction F``: training windows come from :class:`frontierlab.longctx.data.LongDocData`
  (fraction F within-document windows of documents >= ``--seq`` tokens, the rest ordinary windows).
  Omit it to train on the loop's ordinary random windows (which cross document boundaries).
  ``--short-data DIR`` draws the ordinary windows from another prepared folder (long documents in
  ``--data``, Data-v0 in ``--short-data``).

Exact resume is unchanged: rerunning the same command after an interruption loads ``<run>/checkpoint.pt``
(model, optimizer, step, RNG states) on top of the initial weights, so the run continues as the same run.
The run card gets a ``longctx`` block (initial checkpoint, its SHA-256 and step, RoPE rule, data mode)
and ``parent_run`` set to the initial checkpoint's run folder.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

import frontierlab.longctx  # noqa: F401  (registers the attention kinds)
from frontierlab.data.loader import TokenData
from frontierlab.longctx.data import LongDocData
from frontierlab.model import LM, PRESETS, ModelConfig
from frontierlab.train import loop

MODULE4_KINDS = ("gqa-rope-scaled", "gqa-irope")
SIZE_KEYS = ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
             "head_dim", "intermediate_size", "qk_norm", "tie_word_embeddings")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 add_help=False, allow_abbrev=False)
    ap.add_argument("--init-from", type=Path, default=None, help="checkpoint.pt to start from (weights only)")
    ap.add_argument("--rope", default="default", choices=["default", "pi", "ntk", "yarn", "llama3", "proportional"])
    ap.add_argument("--factor", type=float, default=1.0, help="scale s = new length / trained length")
    ap.add_argument("--original", type=int, default=None, help="trained length (YaRN, llama3)")
    ap.add_argument("--beta-fast", type=float, default=32.0)
    ap.add_argument("--beta-slow", type=float, default=1.0)
    ap.add_argument("--ramp", default="code", choices=["code", "paper"])
    ap.add_argument("--attention-factor", type=float, default=None, help="override YaRN's 0.1 ln s + 1")
    ap.add_argument("--theta", type=float, default=None, help="new RoPE base (base-frequency adjustment)")
    ap.add_argument("--partial", type=float, default=None, help="partial_rotary_factor (from-scratch runs only)")
    ap.add_argument("--nope-every", type=int, default=None, help="use gqa-irope with a NoPE layer every N layers")
    ap.add_argument("--irope-temperature", action="store_true")
    ap.add_argument("--floor-scale", type=float, default=8192)
    ap.add_argument("--long-fraction", type=float, default=None,
                    help="fraction of within-document windows from documents >= --seq tokens")
    ap.add_argument("--short-data", type=Path, default=None,
                    help="prepared folder for the ordinary (non-long) windows; default: the loop's --data")
    return ap


def rope_dict(m) -> dict:
    d = {"type": m.rope, "factor": m.factor, "beta_fast": m.beta_fast, "beta_slow": m.beta_slow, "ramp": m.ramp}
    if m.original is not None:
        d["original_max_position_embeddings"] = m.original
    if m.attention_factor is not None:
        d["attention_factor"] = m.attention_factor
    if m.partial is not None:
        d["partial_rotary_factor"] = m.partial
    return d


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model(m, preset: str, vocab_size: int, seq: int):
    """Model config and weights for the extended run (pure function; tested without training)."""
    base = PRESETS[preset](vocab_size=vocab_size)
    init = None
    if m.init_from is not None:
        init = torch.load(m.init_from, map_location="cpu", weights_only=False)
        ck = ModelConfig(**init["config"])
        bad = {k: (getattr(ck, k), getattr(base, k)) for k in SIZE_KEYS if getattr(ck, k) != getattr(base, k)}
        if bad:
            raise ValueError(f"checkpoint config does not match --preset {preset}: {bad}")
        base = ck
        if m.partial is not None and m.partial != ck.extra.get("rope", {}).get("partial_rotary_factor", 1.0):
            raise ValueError("partial RoPE changes which channels carry position; it cannot be switched on a trained model")
    extra = {**base.extra, "rope": rope_dict(m)}
    kind = "gqa-rope-scaled"
    if m.nope_every:
        kind = "gqa-irope"
        extra["irope"] = {"nope_every": m.nope_every, "temperature": m.irope_temperature, "floor_scale": m.floor_scale}
    if init is not None and base.attention == "gqa-irope" and kind != "gqa-irope":
        raise ValueError("the checkpoint was trained with gqa-irope; pass the same --nope-every")
    cfg = base.with_(attention=kind, extra=extra, max_position_embeddings=max(seq, base.max_position_embeddings))
    if m.theta is not None:
        cfg = cfg.with_(rope_theta=m.theta)
    model = LM(cfg)
    if init is not None:
        model.load_state_dict(init["model"], strict=True)
    return cfg, model, init


def main(argv=None):
    mine, rest = build_parser().parse_known_args(argv)
    if any(x in ("-h", "--help") for x in rest):
        print(__doc__)
        return None
    if "--attention" in rest:
        raise SystemExit("use --rope / --nope-every here; --attention is set by this wrapper")
    loop.build_parser().parse_args(rest)          # fail early on a typo, before any work
    info = {"init_from": str(mine.init_from) if mine.init_from else None, "rope": rope_dict(mine),
            "theta": mine.theta, "nope_every": mine.nope_every, "long_fraction": mine.long_fraction,
            "short_data": str(mine.short_data) if mine.short_data else None}
    if mine.init_from is not None:
        info["init_sha256"] = sha256(mine.init_from)

    def make_model(a, vocab_size):
        cfg, model, init = build_model(mine, a.preset, vocab_size, a.seq)
        if getattr(a, "extra", None):                    # the loop's --extra JSON, merged on top
            import json
            cfg = cfg.with_(extra={**cfg.extra, **json.loads(a.extra)})
            sd = model.state_dict()
            model = LM(cfg)
            model.load_state_dict(sd, strict=True)
        if init is not None:
            info["init_step"] = int(init["step"])
        return cfg, model

    def token_data(split, **kw):
        if split == "train" and mine.long_fraction is not None:
            return LongDocData(split, long_fraction=mine.long_fraction, short_root=mine.short_data, **kw)
        return TokenData(split, **kw)

    orig_card = loop.write_run_card

    def write_card(run_dir, **kw):
        kw["extra"] = {**(kw.get("extra") or {}), "longctx": info}
        if kw.get("parent") is None and mine.init_from is not None:
            kw["parent"] = Path(mine.init_from).parent.name
        return orig_card(run_dir, **kw)

    orig_fpt = loop.flops_per_token

    def fpt(cfg, T, *args, **kw):
        # RoPE rules and NoPE layers do not change matmul or attention-score FLOPs, so the Module 4 kinds
        # are counted as GQA (until frontierlab.attention.accounting lists them; see the Module 4 inbox)
        if cfg.attention in MODULE4_KINDS:
            cfg = cfg.with_(attention="gqa", extra={})
        return orig_fpt(cfg, T, *args, **kw)

    saved = loop.make_model, loop.TokenData, loop.write_run_card, loop.flops_per_token
    loop.make_model, loop.TokenData, loop.write_run_card, loop.flops_per_token = make_model, token_data, write_card, fpt
    try:
        return loop.main(rest)
    finally:
        loop.make_model, loop.TokenData, loop.write_run_card, loop.flops_per_token = saved


if __name__ == "__main__":
    main()
