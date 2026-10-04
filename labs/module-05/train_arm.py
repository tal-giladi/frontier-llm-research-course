"""Train one Module 5 arm with the course training loop (``frontierlab.train.loop``), unchanged.

    python labs/module-05/train_arm.py --arm hybrid-kda -- --run runs/m05/l51/hybrid-kda --preset toy \\
        --steps 600 --batch 16 --seq 256
    python labs/module-05/train_arm.py --arm dsa --init-from runs/m04/base-cpu/checkpoint.pt --dsa-stage warmup \\
        -- --run runs/m05/l52/warmup --preset toy --steps 150 --seq 256 --lr 1e-3
    python labs/module-05/train_arm.py --arm dsa --init-from runs/m05/l52/warmup/checkpoint.pt --dsa-stage sparse \\
        -- --run runs/m05/l52/sparse --preset toy --steps 300 --seq 256 --lr 3e-4

Everything after ``--`` goes to the loop as it is. This wrapper:

1. builds the arm's config (``m05.arm_config``; ``--match-params`` for the equal-parameters axis,
   ``--topk`` to override DSA's k, ``--chunk`` for the linear kinds' chunk size, ``--linear-mode fla``
   for the flash-linear-attention kernels on the GPU path);
2. optionally starts from the weights of a checkpoint (``--init-from``; for a DSA arm from a dense run the
   indexer starts fresh: it is the only missing parameter, and the wrapper checks that);
3. for DSA, implements the two documented training stages (DeepSeek-V3.2 section 2.1.1):

   * ``--dsa-stage warmup``: dense attention, every parameter frozen except the indexer, the indexer
     trained by its KL loss to the head-summed, L1-normalised main attention;
   * ``--dsa-stage sparse``: top-k selection on; the indexer (whose input is detached) trained only by its
     KL loss over the selected set, the main model only by the LM loss.

   The loop's training loss stays the LM loss: the wrapper returns ``lm + (L_I - L_I.detach())``, whose
   value is the LM loss and whose gradient also contains L_I's (which reaches only the indexer). L_I and
   the indexer's attention-mass recall are logged to ``<run>/indexer.jsonl``.
4. makes the loop's run card and MFU use ``accounting.m05_flops_per_token`` (exact for every kind; the
   loop's ``accounting.flops_per_token`` raises for the Module 5 kinds), and records the arm, stage and
   parent checkpoint in the run card.

Exact resume works as in lesson 01.1: rerun the same command.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m05  # noqa: E402
from frontierlab.attention import accounting  # noqa: E402
from frontierlab.attention import dsa as dsa_mod  # noqa: E402
from frontierlab.model import LM  # noqa: E402
from frontierlab.train import loop  # noqa: E402


class DSATrainLM(LM):
    """LM whose training loss adds the indexer objective with value-preserving bookkeeping."""

    stage = "sparse"
    log_path: Path | None = None
    _step = 0

    def forward(self, idx, labels=None, cache=None, reduction="mean"):
        collect = self.training and labels is not None and cache is None
        dsa_mod.set_dsa(self, collect=collect)
        out = super().forward(idx, labels=labels, cache=cache, reduction=reduction)
        dsa_mod.set_dsa(self, collect=False)
        if not collect:
            return out
        li = dsa_mod.indexer_loss(self)
        lm = out.loss if self.stage == "sparse" else out.loss.detach()
        out.loss = lm + (li - li.detach())
        if self.log_path is not None:
            self._step += 1
            rec = [m.last_recall for m in dsa_mod.dsa_layers(self) if m.last_recall is not None]
            row = {"micro_step": self._step, "indexer_kl": float(li.detach()), "stage": self.stage}
            if rec:
                for key in ("indexer", "oracle", "window"):
                    row[f"recall_{key}"] = sum(r[key] for r in rec) / len(rec)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        for m in dsa_mod.dsa_layers(self):
            m.indexer_loss = None
        return out


def build(arm, a, vocab, w):
    base = m05.preset_config(a.preset, vocab)
    cfg = m05.arm_config(arm, base, a.seq, match_params=w.match_params, chunk=w.chunk, topk=w.topk,
                         linear_mode=w.linear_mode)
    if w.dsa_stage == "warmup":
        cfg = cfg.with_(extra={**cfg.extra, "dsa_mode": "dense"})
    if getattr(a, "attention", None):          # the loop's --attention: a variant of the arm's kind
        cfg = cfg.with_(attention=a.attention)
    if getattr(a, "extra", None):
        cfg = cfg.with_(extra={**cfg.extra, **json.loads(a.extra)})
    cls = DSATrainLM if arm == "dsa" else LM
    model = cls(cfg)
    if w.init_from:
        ck = torch.load(w.init_from, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        bad = [n for n in missing if ".idx_" not in n]
        if unexpected or bad:
            raise SystemExit(f"--init-from does not fit this arm: missing {bad[:5]}, unexpected {unexpected[:5]}")
    if arm == "dsa":
        model.stage = w.dsa_stage or "sparse"
        if model.stage == "warmup":
            idx = {id(p) for p in dsa_mod.indexer_parameters(model)}
            for p in model.parameters():
                p.requires_grad_(id(p) in idx)
    return cfg, model


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        raise SystemExit("usage: train_arm.py --arm ARM [options] -- <loop args>")
    i = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=m05.ARMS, required=True)
    ap.add_argument("--match-params", action="store_true")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--linear-mode", choices=["chunked", "recurrent", "fla"], default="chunked")
    ap.add_argument("--init-from", type=Path, default=None)
    ap.add_argument("--dsa-stage", choices=["warmup", "sparse"], default=None)
    w = ap.parse_args(argv[:i])
    rest = argv[i + 1:]
    a = loop.build_parser().parse_args(rest)
    info = {"arm": w.arm, "match_params": w.match_params, "dsa_stage": w.dsa_stage,
            "init_from": str(w.init_from) if w.init_from else None}

    def make_model(args, vocab_size):
        cfg, model = build(w.arm, args, vocab_size, w)
        if isinstance(model, DSATrainLM):
            args.run.mkdir(parents=True, exist_ok=True)
            model.log_path = args.run / "indexer.jsonl"
        print(f"arm {w.arm}{' / ' + w.dsa_stage if w.dsa_stage else ''}: {m05.describe(cfg, args.seq)}")
        return cfg, model

    orig_card = loop.write_run_card

    def write_card(run_dir, **kw):
        kw["extra"] = {**(kw.get("extra") or {}), "m05": info}
        if kw.get("parent") is None and w.init_from is not None:
            kw["parent"] = Path(w.init_from).parent.name
        return orig_card(run_dir, **kw)

    saved = loop.make_model, loop.flops_per_token, loop.param_counts, loop.write_run_card
    loop.make_model = make_model
    loop.flops_per_token = accounting.m05_flops_per_token
    loop.param_counts = lambda c: {k: v for k, v in accounting.param_counts(c).items()
                                   if k in ("total", "non_embedding", "embedding")}
    loop.write_run_card = write_card
    try:
        return loop.main(rest)
    finally:
        loop.make_model, loop.flops_per_token, loop.param_counts, loop.write_run_card = saved


if __name__ == "__main__":
    main()
