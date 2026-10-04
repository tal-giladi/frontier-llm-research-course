"""Module 6 training runs: the course loop with a BlockLM (MTP, HC/mHC, MoE, Engram, MatFormer, PLE, diffusion).

    python -m frontierlab.blocks.train --mtp deepseek --mtp-depth 1 --mtp-schedule deepseek \\
        --run runs/m06/mtp --preset toy --steps 200 --batch 16 --seq 128 --lr 1.5e-3

Every argument the loop knows is passed to ``frontierlab.train.loop`` unchanged (``--attention`` and ``--extra``
too, so Module 3–5 attention kinds combine with these switches). This wrapper replaces ``loop.make_model``,
``loop.make_optimizer``, ``loop.lr_at``, ``loop.param_counts``, ``loop.flops_per_token`` and
``loop.write_run_card`` for the duration of one ``loop.main()`` call (the pattern of
``frontierlab.longctx.extend`` and ``frontierlab.optim.train``); the native loop changes are proposed in
``curriculum/inbox/module-06-shared-changes.md``. What it adds:

* model switches (all stored in ``cfg.extra["blocks"]``, hence in the run card and the checkpoint):
  ``--mtp {none,meta,deepseek} --mtp-depth D --mtp-lambda L --mtp-schedule {constant,deepseek}``
  (``deepseek``: 0.3 for the first 10/14.8 of the run, then 0.1, DeepSeek-V3 section 4.2);
  ``--residual {plain,hc,mhc} --streams n --sinkhorn-iters t``;
  ``--ffn {dense,moe,matformer}`` with ``--moe-experts E --moe-top-k k --moe-intermediate I --moe-shared s
  --moe-aux c --moe-first-dense f``; ``--engram-layers 1 3 --engram-table M --engram-heads K --engram-max-n N``;
  ``--ple-dim d``; ``--cfg intermediate_size=48 num_hidden_layers=3`` (preset fields, e.g. Meta's
  parameter matching or a Baseline-0 scaled to a budget); ``--blocks JSON`` (a whole ``blocks`` dict); ``--objective diffusion`` (DiffusionLM with
  ``"gqa-bidir"`` attention and one extra vocabulary id for the mask token).
* parameter groups: AdamW as in the loop, but no weight decay on the static HC/mHC mappings and gating
  factors (decay would pull H_res towards 0 and break the identity path; HC section 5 "The static
  component ... does not utilize weight decay") and on Engram tables (Engram section 4).
* exact accounting (``frontierlab.blocks.accounting``) in ``budget.train_flops``, ``params`` and MFU.
* ``--blocks-log [--hyper-every N]``: ``<run>/blocks.jsonl`` every ``--log-every`` steps with the main and MTP
  losses, lambda, MoE balance loss and max load, and every N steps the HC/mHC Amax gains (mHC section 3.1).

Exact resume holds (MatFormer's granularity draws come from a step counter saved in the state dict; the
diffusion masks from torch's RNG, which the loop saves). Rerun the same command after an interruption.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from frontierlab.blocks import accounting
from frontierlab.blocks.model import BlockLM
from frontierlab.blocks.mtp import DEEPSEEK_V3_LAMBDA, lambda_at
from frontierlab.metrics.jsonl import JsonlLogger
from frontierlab.model import PRESETS
from frontierlab.train import loop

NO_DECAY_KEYS = ("b_static", "am_static", "ar_static", "s_alpha", "s_beta", "b_pre", "b_post", "b_res",
                 "alpha_pre", "alpha_post", "alpha_res", ".tables.", "norm")


def build_parser():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--mtp", choices=["none", "meta", "deepseek"], default=None)
    ap.add_argument("--mtp-depth", type=int, default=None)
    ap.add_argument("--mtp-lambda", type=float, default=None)
    ap.add_argument("--mtp-schedule", choices=["constant", "deepseek"], default="constant")
    ap.add_argument("--residual", choices=["plain", "hc", "mhc"], default=None)
    ap.add_argument("--streams", type=int, default=None)
    ap.add_argument("--sinkhorn-iters", type=int, default=None)
    ap.add_argument("--ffn", choices=["dense", "moe", "matformer"], default=None)
    ap.add_argument("--moe-experts", type=int, default=None)
    ap.add_argument("--moe-top-k", type=int, default=None)
    ap.add_argument("--moe-intermediate", type=int, default=None)
    ap.add_argument("--moe-shared", type=int, default=None)
    ap.add_argument("--moe-aux", type=float, default=None)
    ap.add_argument("--moe-first-dense", type=int, default=None)
    ap.add_argument("--engram-layers", type=int, nargs="*", default=None)
    ap.add_argument("--engram-table", type=int, default=None)
    ap.add_argument("--engram-heads", type=int, default=None)
    ap.add_argument("--engram-max-n", type=int, default=None)
    ap.add_argument("--ple-dim", type=int, default=None)
    ap.add_argument("--blocks", default=None, help="JSON dict merged into cfg.extra['blocks'] first")
    ap.add_argument("--objective", choices=["ar", "diffusion"], default="ar")
    ap.add_argument("--cfg", nargs="*", default=[], metavar="FIELD=VALUE",
                    help="override ModelConfig fields of the preset, e.g. intermediate_size=48 num_hidden_layers=3")
    ap.add_argument("--blocks-log", action="store_true")
    ap.add_argument("--hyper-every", type=int, default=50)
    return ap


def blocks_dict(m) -> dict:
    """The ``blocks`` settings implied by the command line (only what was given)."""
    b = json.loads(m.blocks) if m.blocks else {}
    for key, val in (("mtp", m.mtp), ("mtp_depth", m.mtp_depth), ("mtp_lambda", m.mtp_lambda),
                     ("residual", m.residual), ("streams", m.streams), ("sinkhorn_iters", m.sinkhorn_iters),
                     ("ffn", m.ffn)):
        if val is not None:
            b[key] = val
    moe = {k: v for k, v in (("experts", m.moe_experts), ("top_k", m.moe_top_k), ("intermediate", m.moe_intermediate),
                             ("shared", m.moe_shared), ("aux_coef", m.moe_aux), ("first_dense", m.moe_first_dense))
           if v is not None}
    if moe:
        b["moe"] = {**b.get("moe", {}), **moe}
    eng = {k: v for k, v in (("layers", m.engram_layers), ("table", m.engram_table), ("heads", m.engram_heads),
                             ("max_n", m.engram_max_n)) if v is not None}
    if eng:
        b["engram"] = {**(b.get("engram") or {}), **eng}
    if m.ple_dim is not None:
        b["ple"] = {"dim": m.ple_dim}
    return b


def model_config(m, a, vocab_size: int):
    """The run's ModelConfig: preset, loop's --attention / --extra, then the blocks settings."""
    cfg = PRESETS[a.preset](vocab_size=vocab_size + (1 if m.objective == "diffusion" else 0))
    for item in m.cfg:
        key, val = item.split("=", 1)
        if not hasattr(cfg, key) or key in ("extra", "attention"):
            raise SystemExit(f"--cfg: unknown or reserved field {key!r}")
        cfg = cfg.with_(**{key: type(getattr(cfg, key))(float(val) if "." in val or "e" in val else int(val))
                           if not isinstance(getattr(cfg, key), bool) else val.lower() in ("1", "true", "on")})
    if a.attention:
        cfg = cfg.with_(attention=a.attention)
    if m.objective == "diffusion":
        cfg = cfg.with_(attention="gqa-bidir")
    extra = dict(cfg.extra)
    if a.extra:
        extra.update(json.loads(a.extra))
    extra["blocks"] = {**extra.get("blocks", {}), **blocks_dict(m), "seed": a.seed}
    return cfg.with_(extra=extra)


def build_model(cfg, objective: str = "ar"):
    if objective == "diffusion" or cfg.attention == "gqa-bidir":
        from frontierlab.blocks.diffusion import DiffusionLM
        return DiffusionLM(cfg)
    return BlockLM(cfg)


def load_model(checkpoint, map_location="cpu"):
    """Rebuild any Module 6 model (BlockLM or DiffusionLM) from a loop checkpoint."""
    from frontierlab.model import ModelConfig
    ck = torch.load(checkpoint, map_location=map_location, weights_only=False)
    cfg = ModelConfig(**ck["config"])
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    return model


def param_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.dim() < 2 or any(k in n for k in NO_DECAY_KEYS) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def main(argv=None):
    mine, rest = build_parser().parse_known_args(argv)
    if any(x in ("-h", "--help") for x in rest):
        print(__doc__)
        return None
    a0 = loop.build_parser().parse_args(rest)               # fail early on a typo
    if a0.loss == "chunked" and (mine.mtp not in (None, "none") or mine.objective == "diffusion"):
        raise SystemExit("--loss chunked computes only the next-token loss; it would drop the MTP / diffusion loss")
    if a0.compile and mine.blocks_log:
        raise SystemExit("--compile with --blocks-log is not supported (the log reads Python-side stats)")
    holder: dict = {"step": 0}
    info = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(mine).items()}

    def make_model(a, vocab_size):
        cfg = model_config(mine, a, vocab_size)
        model = build_model(cfg, mine.objective)
        holder["model"], holder["a"] = model, a
        return cfg, model

    def make_optimizer(model, a):
        opt = torch.optim.AdamW(param_groups(model, a.weight_decay), lr=a.lr, betas=(0.9, 0.95),
                                fused=a.device.startswith("cuda"))
        if mine.blocks_log:
            log = holder["log"] = JsonlLogger(a.run / "blocks.jsonl")
            orig_step = opt.step

            def step(*args, **kw):
                out = orig_step(*args, **kw)
                s = holder["step"] + 1
                if s % a.log_every == 0 or s == a.steps:
                    st = dict(getattr(model, "last_stats", {}))
                    st.pop("mtp_losses", None)
                    if getattr(model, "blocks", {}).get("mtp", "none") != "none":
                        st["mtp_lambda"] = model.mtp_lambda
                    log.log(step=s, **st)
                return out
            opt.step = step
        return opt

    def lr_at(step, steps, lr, warmup, schedule="cosine", *args, **kw):
        model = holder.get("model")
        holder["step"] = step
        if model is not None:
            if mine.mtp_schedule == "deepseek" and isinstance(model, BlockLM):
                model.mtp_lambda = lambda_at(step, steps, DEEPSEEK_V3_LAMBDA)
            if mine.blocks_log and hasattr(model, "track_hyper"):
                model.track_hyper = (step + 1) % mine.hyper_every == 0 or step + 1 == steps
        return saved[2](step, steps, lr, warmup, schedule, *args, **kw)

    def pcounts(cfg):
        return accounting.param_counts(cfg)

    def fpt(cfg, T, *args, **kw):
        return accounting.flops_per_token(cfg, T, *args, **kw)

    def write_card(run_dir, **kw):
        kw["extra"] = {**(kw.get("extra") or {}), "blocks": info}
        return saved[5](run_dir, **kw)

    names = ("make_model", "make_optimizer", "lr_at", "param_counts", "flops_per_token", "write_run_card")
    saved = [getattr(loop, n) for n in names]
    for n, f in zip(names, (make_model, make_optimizer, lr_at, pcounts, fpt, write_card)):
        setattr(loop, n, f)
    try:
        return loop.main(rest)
    finally:
        for n, f in zip(names, saved):
            setattr(loop, n, f)
        if "log" in holder:
            holder["log"].close()


if __name__ == "__main__":
    main()
