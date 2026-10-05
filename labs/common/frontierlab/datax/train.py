"""Module 10 training runs: the course loop on a data mixture, with document masking, continued training
and anneals. A wrapper around ``frontierlab.train.loop`` (pattern: ``frontierlab.longctx.extend``).

    python -m frontierlab.datax.train --mixture runs/m10/mix.json --doc-mask \\
        --run runs/m10/mix-a --preset toy --steps 300 --batch 16 --seq 128

    # micro-anneal / continued training from a checkpoint (weights only, fresh optimizer)
    python -m frontierlab.datax.train --init-from runs/m10/stable/checkpoint.pt --anneal \\
        --mixture runs/m10/anneal-math.json --run runs/m10/anneal-math --preset toy --steps 100 --warmup 10

Every argument the loop knows is passed through unchanged. Added here (the native loop flags are
proposed in ``curriculum/inbox/module-10-shared-changes.md``):

* ``--mixture SPEC.json``: training windows come from :class:`frontierlab.datax.mixture.MixtureSampler`
  (deterministic, packed, exact per-source token accounting). Validation stays the loop's
  ``TokenData("val", --data)`` (Eval v0's split). Resume: the sampler is a function of the window
  counter, so on restart it seeks to ``step × grad_accum × batch`` read from ``<run>/checkpoint.pt``.
* ``--doc-mask``: attention kind ``"gqa-docmask"`` (block-diagonal causal attention by document,
  :mod:`frontierlab.datax.packing`), and the loop's validation uses the same mask. Without
  ``--mixture`` the training windows are the loop's own random windows (same generator, same order),
  with segment ids computed from ``train_docs.npy``, so a masked and an unmasked run see identical
  tokens: the matched ablation of lesson 10.1.
* ``--init-from CKPT``: start from that checkpoint's weights (config from the checkpoint, which must
  match ``--preset``); optimizer, step and schedule start fresh. Ignored when ``<run>/checkpoint.pt``
  exists (exact resume wins).
* ``--anneal``: after warmup, decay the learning rate linearly from ``--lr`` to zero over the rest of
  the run (OLMo 2's mid-training schedule, section 4.1; the micro-anneals of lesson 10.4).

The run card gets a ``datax`` block: the mixture spec and its digest, the **planned** per-source token
accounting for the whole run (known before step 1), the document-mask flag and the initial checkpoint
(path, SHA-256, step). ``<run>/mixture_accounting.json`` holds the accounting of what was consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from frontierlab.data.loader import TokenData
from frontierlab.datax import packing
from frontierlab.datax.mixture import MixtureSampler, MixtureSpec
from frontierlab.model import LM, PRESETS, ModelConfig
from frontierlab.train import loop

SIZE_KEYS = ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
             "head_dim", "intermediate_size", "qk_norm", "tie_word_embeddings")
GQA_COMPATIBLE = ("gqa", "gqa-docmask")


def build_parser():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--mixture", type=Path, default=None)
    ap.add_argument("--doc-mask", action="store_true")
    ap.add_argument("--init-from", type=Path, default=None)
    ap.add_argument("--anneal", action="store_true")
    return ap


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class SegmentedTokenData(TokenData):
    """The loop's random windows (identical generator use), plus document segments for the mask."""

    def batch(self, B, T, generator, device="cpu"):
        starts = torch.randint(0, len(self.tokens) - T, (B,), generator=generator).tolist()
        x = torch.stack([self.window(s, T) for s in starts]).to(device)
        seg = torch.from_numpy(np.stack([packing.segment_ids(self.doc_starts, s, T) for s in starts])).to(device)
        if getattr(self, "_cm", None) is not None:
            self._cm.__exit__(None, None, None)
        self._cm = packing.document_segments(seg)
        self._cm.__enter__()
        return x

    def close(self):
        if getattr(self, "_cm", None) is not None:
            self._cm.__exit__(None, None, None)
            self._cm = None


def anneal_lr(step: int, steps: int, lr: float, warmup: int, *args, **kw) -> float:
    """Linear warmup, then linear decay from ``lr`` to 0 at ``steps`` (the last step trains at lr/(steps-warmup))."""
    if step < warmup:
        return lr * (step + 1) / warmup
    return lr * max(0.0, (steps - step) / max(1, steps - warmup))


def load_init(path: Path, preset: str, vocab_size: int):
    init = torch.load(path, map_location="cpu", weights_only=False)
    ck = ModelConfig(**init["config"])
    base = PRESETS[preset](vocab_size=vocab_size)
    bad = {k: (getattr(ck, k), getattr(base, k)) for k in SIZE_KEYS if getattr(ck, k) != getattr(base, k)}
    if bad:
        raise ValueError(f"--init-from checkpoint does not match --preset {preset}: {bad}")
    return ck, init


def main(argv=None):
    mine, rest = build_parser().parse_known_args(argv)
    if any(x in ("-h", "--help") for x in rest):
        print(__doc__)
        return None
    a = loop.build_parser().parse_args(rest)          # fail early on a typo; read batch/seq/run
    if mine.doc_mask and a.attention not in (None, "gqa", "gqa-docmask"):
        raise SystemExit("--doc-mask is implemented for Baseline-0's GQA attention only")
    spec = MixtureSpec.load(mine.mixture) if mine.mixture else None
    info = {"mixture": spec.to_dict() if spec else None, "mixture_digest": spec.digest() if spec else None,
            "doc_mask": mine.doc_mask, "anneal": mine.anneal,
            "init_from": str(mine.init_from) if mine.init_from else None}
    if mine.init_from:
        info["init_sha256"] = sha256(mine.init_from)
    windows_per_step = a.batch * a.grad_accum
    resume_step = 0
    ckpt = a.run / "checkpoint.pt"
    if ckpt.exists():
        resume_step = int(torch.load(ckpt, map_location="cpu", weights_only=False)["step"])
    holder: dict = {}

    def make_model(args, vocab_size):
        if mine.init_from is not None:
            cfg, init = load_init(mine.init_from, args.preset, vocab_size)
            info["init_step"] = int(init["step"])
            if cfg.attention not in GQA_COMPATIBLE:
                raise ValueError(f"--init-from with attention {cfg.attention!r}: Module 10 runs use gqa or gqa-docmask")
        else:
            cfg, init = PRESETS[args.preset](vocab_size=vocab_size), None
            if args.attention:
                cfg = cfg.with_(attention=args.attention)
        if mine.doc_mask:
            cfg = cfg.with_(attention="gqa-docmask")
        elif cfg.attention == "gqa-docmask":
            cfg = cfg.with_(attention="gqa")
        if args.extra:
            cfg = cfg.with_(extra={**cfg.extra, **json.loads(args.extra)})
        model = LM(cfg)
        if init is not None:
            model.load_state_dict(init["model"], strict=True)
        return cfg, model

    def token_data(split, **kw):
        if split == "train":
            if spec is not None:
                s = MixtureSampler(spec, doc_mask=mine.doc_mask)
                s.seek(resume_step * windows_per_step)
                holder["train"] = s
                return s
            if mine.doc_mask:
                holder["train"] = SegmentedTokenData(split, **kw)
                return holder["train"]
        return TokenData(split, **kw)

    orig_card = loop.write_run_card

    def write_card(run_dir, **kw):
        extra = dict(kw.get("extra") or {})
        if spec is not None:
            info["planned_accounting"] = holder["train"].accounting(a.steps * windows_per_step, a.seq)
        extra["datax"] = info
        kw["extra"] = extra
        if kw.get("parent") is None and mine.init_from is not None:
            kw["parent"] = Path(mine.init_from).parent.name
        if spec is not None:                         # the data block names the mixture, not Data-v0 alone
            kw["data_meta"] = {**(kw.get("data_meta") or {}), "name": f"mixture:{spec.name}@{spec.digest()}"}
        return orig_card(run_dir, **kw)

    orig_fpt = loop.flops_per_token

    def fpt(cfg, T, *args, **kw):                    # the mask does not change SDPA's FLOPs (see packing.py)
        if cfg.attention == "gqa-docmask":
            cfg = cfg.with_(attention="gqa")
        return orig_fpt(cfg, T, *args, **kw)

    orig_wl = loop.window_losses

    def window_losses(model, data, n, T, **kw):
        if mine.doc_mask:
            with packing.document_segments(None):     # the training batch's segments must not leak in
                return packing.window_losses_docmask(model, data, n, T, **kw)
        return orig_wl(model, data, n, T, **kw)

    names = ("make_model", "TokenData", "write_run_card", "flops_per_token", "window_losses", "lr_at")
    saved = {n: getattr(loop, n) for n in names}
    loop.make_model, loop.TokenData, loop.write_run_card = make_model, token_data, write_card
    loop.flops_per_token, loop.window_losses = fpt, window_losses
    if mine.anneal:
        loop.lr_at = anneal_lr
    try:
        model = loop.main(rest)
    finally:
        for n, v in saved.items():
            setattr(loop, n, v)
        tr = holder.get("train")
        if tr is not None and hasattr(tr, "close"):
            tr.close()
    if spec is not None:
        acc = holder["train"].accounting(holder["train"].k, a.seq)
        (a.run / "mixture_accounting.json").write_text(json.dumps(acc, indent=2))
    return model


if __name__ == "__main__":
    main()
