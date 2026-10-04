"""Module 7 training runs: the course loop with another optimizer, parametrization, schedule or stabilizer.

    python -m frontierlab.optim.train --optimizer muon --stability-log \\
        --run runs/m07/muon --preset toy --steps 300 --batch 16 --seq 128 --lr 3e-3

Every argument the loop knows is passed to ``frontierlab.train.loop`` unchanged. This wrapper replaces
``loop.make_model``, ``loop.make_optimizer``, ``loop.lr_at``, ``loop.TokenData``, ``loop.write_run_card`` and
``loop.flops_per_token`` for the duration of one ``loop.main()`` call (like ``frontierlab.longctx.extend``);
the native loop changes are proposed in ``curriculum/inbox/module-07-loop-changes.md``. What it adds:

* optimizer: ``--optimizer {adamw-torch,adamw,muon}``. ``adamw-torch`` is the loop's own AdamW, untouched;
  ``adamw`` and ``muon`` use :class:`frontierlab.optim.muon.MuonAdamW` (so both arms of a Muon-vs-AdamW
  comparison run the same code). Muon options: ``--muon-adjust {match_rms,original,none}``,
  ``--momentum``, ``--no-nesterov``, ``--ns {quintic5,v4-hybrid,cubic5,cubic20}``, ``--ns-dtype``.
* ``--qk-clip TAU``: QK-Clip after every step (``frontierlab.optim.qkclip``); needs ``--qk-norm off`` or MLA.
* model: ``--qk-norm {on,off}``, ``--width W`` (the preset widened by ``mup.width_config``),
  ``--mup-base-width W0`` (µP init, readout multiplier and per-group learning rates relative to W0),
  ``--mup-muon-rule {spectral,adam}``, ``--z-loss C``, ``--final-softcap C``, ``--attn-softcap C``,
  ``--clip-qkv C`` (the last two switch attention to ``"gqa-softcap"``).
* schedule: ``--decay-start S0 --decay-steps D --decay-shape {linear,1-sqrt,cosine} --min-lr-ratio r`` for
  ``--schedule wsd`` (``frontierlab.optim.schedules.lr_at``); ``--branch-from CKPT`` starts a run from a full
  checkpoint (weights, optimizer, step, RNG) of another run, so a decay branch continues the stable run;
  ``--init-from CKPT`` takes only the weights (fresh optimizer, step 0: continued training with a re-warm).
* ``--stability-log [--stability-every N]``: ``<run>/stability.jsonl`` (``frontierlab.optim.stability``).
* ``--inject-bad-steps 120,121``: on those steps every training window is replaced by a repeated short
  token pattern (the kind of data OLMo 2 filters out, arXiv 2501.00656 section 3.1), for lesson 07.5.

The run card gets an ``optim`` block with these settings and ``budget.optimizer_flops`` (Newton-Schulz FLOPs
for the whole run). Exact resume holds: rerun the same command after an interruption.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from frontierlab.data.loader import TokenData
from frontierlab.model import PRESETS, ModelConfig
from frontierlab.optim import cost, mup, schedules
from frontierlab.optim.muon import MuonAdamW, param_groups
from frontierlab.optim.qkclip import QKClip
from frontierlab.optim.stability import StabilityLogger
from frontierlab.optim.stabilizers import OptLM
from frontierlab.train import loop

GQA_LIKE = ("gqa-softcap",)


def build_parser():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--optimizer", choices=["adamw-torch", "adamw", "muon"], default="adamw-torch")
    ap.add_argument("--muon-adjust", choices=["match_rms", "original", "none"], default="match_rms")
    ap.add_argument("--momentum", type=float, default=0.95)
    ap.add_argument("--no-nesterov", action="store_true")
    ap.add_argument("--ns", default="quintic5", choices=["quintic5", "v4-hybrid", "cubic5", "cubic20"])
    ap.add_argument("--ns-dtype", default="bfloat16", choices=["bfloat16", "float32", "float64"])
    ap.add_argument("--qk-clip", type=float, default=None, metavar="TAU")
    ap.add_argument("--qk-norm", choices=["on", "off"], default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--mup-base-width", type=int, default=None)
    ap.add_argument("--mup-muon-rule", choices=["spectral", "adam"], default="spectral")
    ap.add_argument("--z-loss", type=float, default=None)
    ap.add_argument("--final-softcap", type=float, default=None)
    ap.add_argument("--attn-softcap", type=float, default=None)
    ap.add_argument("--clip-qkv", type=float, default=None)
    ap.add_argument("--decay-start", type=int, default=None)
    ap.add_argument("--decay-steps", type=int, default=None)
    ap.add_argument("--decay-shape", choices=list(schedules.SHAPES), default="linear")
    ap.add_argument("--min-lr-ratio", type=float, default=0.0)
    ap.add_argument("--branch-from", type=Path, default=None)
    ap.add_argument("--init-from", type=Path, default=None)
    ap.add_argument("--stability-log", action="store_true")
    ap.add_argument("--stability-every", type=int, default=1)
    ap.add_argument("--inject-bad-steps", default=None)
    return ap


def model_config(m, preset: str, vocab_size: int, attention: str | None = None, extra_json: str | None = None) -> ModelConfig:
    """The run's config: preset, optional width, QK-norm switch and stabilizer settings in ``extra``."""
    import json
    cfg = PRESETS[preset](vocab_size=vocab_size)
    if m.width:
        cfg = mup.width_config(cfg, m.width)
    if attention:
        cfg = cfg.with_(attention=attention)
    extra = dict(cfg.extra)
    if extra_json:
        extra.update(json.loads(extra_json))
    if m.qk_norm is not None:
        cfg = cfg.with_(qk_norm=m.qk_norm == "on")
    for key, val in (("z_loss", m.z_loss), ("final_softcap", m.final_softcap), ("attn_softcap", m.attn_softcap),
                     ("clip_qkv", m.clip_qkv)):
        if val is not None:
            extra[key] = val
    if m.attn_softcap is not None or m.clip_qkv is not None:
        if cfg.attention != "gqa":
            raise ValueError("--attn-softcap / --clip-qkv are implemented for Baseline-0's GQA only")
        cfg = cfg.with_(attention="gqa-softcap")
    if m.mup_base_width:
        extra["mup"] = {"base_width": m.mup_base_width, "width": cfg.hidden_size}
    return cfg.with_(extra=extra)


def build_model(cfg: ModelConfig):
    """``OptLM(cfg)`` with µP applied if ``cfg.extra["mup"]`` is set (same RNG draws as the run)."""
    model = OptLM(cfg)
    if "mup" in cfg.extra:
        mup.apply_mup(model, cfg.extra["mup"]["base_width"])
    return model


def load_model(checkpoint, map_location="cpu"):
    """Rebuild a Module 7 model from a checkpoint (keeps µP readout and soft-caps; plain LM would drop them)."""
    ck = torch.load(checkpoint, map_location=map_location, weights_only=False)
    cfg = ModelConfig(**ck["config"])
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    return model


def bad_batch(B: int, T: int, period: int = 4, base: int = 100, device="cpu") -> torch.Tensor:
    """(B, T) windows made of one short token pattern repeated: 100 101 102 103 100 101 ..."""
    row = torch.arange(T) % period + base
    return row.expand(B, T).clone().to(device)


def main(argv=None):
    mine, rest = build_parser().parse_known_args(argv)
    if any(x in ("-h", "--help") for x in rest):
        print(__doc__)
        return None
    a0 = loop.build_parser().parse_args(rest)               # fail early on a typo
    if a0.compile and (mine.qk_clip or mine.stability_log):
        raise SystemExit("--compile with --qk-clip/--stability-log is not supported (forward hooks)")
    if a0.loss == "chunked" and (mine.mup_base_width or mine.final_softcap or mine.z_loss):
        raise SystemExit("--loss chunked bypasses the µP readout multiplier, soft-cap and z-loss")
    if mine.optimizer == "adamw-torch" and (mine.mup_base_width or mine.qk_clip or mine.stability_log):
        raise SystemExit("µP, QK-Clip and the stability log need --optimizer adamw or muon")
    if mine.branch_from is not None:
        dst = a0.run / "checkpoint.pt"
        if not dst.exists():
            a0.run.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(mine.branch_from, dst)
            print(f"branching from {mine.branch_from}")
    bad_steps = {int(s) for s in mine.inject_bad_steps.split(",")} if mine.inject_bad_steps else set()
    holder: dict = {}
    info = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(mine).items()}

    def make_model(a, vocab_size):
        cfg = model_config(mine, a.preset, vocab_size, a.attention, a.extra)
        model = build_model(cfg)
        if mine.init_from is not None:                     # weights only; a resume checkpoint overrides them later
            init = torch.load(mine.init_from, map_location="cpu", weights_only=False)
            model.load_state_dict(init["model"], strict=True)
            info["init_step"] = int(init["step"])
        holder["model"] = model
        return cfg, model

    def make_optimizer(model, a):
        if mine.optimizer == "adamw-torch":
            opt = holder["opt"] = saved[1](model, a)
            return opt
        scales = None
        if mine.mup_base_width:
            scales = mup.lr_scales(model, mine.mup_base_width, mine.optimizer, mine.mup_muon_rule, mine.muon_adjust)
        groups = param_groups(model, optimizer=mine.optimizer, lr=a.lr, weight_decay=a.weight_decay,
                              adjust=mine.muon_adjust, momentum=mine.momentum, nesterov=not mine.no_nesterov,
                              schedule=mine.ns, ns_dtype=mine.ns_dtype, lr_scales=scales)
        opt = MuonAdamW(groups, lr=a.lr)
        clip = QKClip(model, mine.qk_clip).attach(opt) if mine.qk_clip else None
        if mine.stability_log:
            holder["stab"] = StabilityLogger(model, opt, a.run / "stability.jsonl", mine.stability_every, qkclip=clip)
        holder["opt"] = opt
        flops = cost.optimizer_flops(model, mine.ns)["total"] if mine.optimizer == "muon" else 0
        info["optimizer_flops_per_step"] = flops
        info["muon_matrices"] = sum(len(g["params"]) for g in groups if g["kind"] == "muon")
        return opt

    def lr_at(step, steps, lr, warmup, schedule="cosine", *args, **kw):
        return schedules.lr_at(step, steps, lr, warmup, schedule, decay_start=mine.decay_start,
                               decay_steps=mine.decay_steps, shape=mine.decay_shape, min_ratio=mine.min_lr_ratio)

    class Data(TokenData):
        def batch(self, B, T, generator, device="cpu"):
            x = super().batch(B, T, generator, device)          # always draw: the data stream stays aligned
            opt = holder.get("opt")
            if bad_steps and self.split == "train" and opt is not None and _steps(opt) + 1 in bad_steps:
                x = bad_batch(B, T, device=x.device)
            return x

    def write_card(run_dir, **kw):
        budget = dict(kw.get("budget") or {})
        budget["optimizer_flops"] = info.get("optimizer_flops_per_step", 0) * budget.get("steps", 0)
        kw["budget"] = budget
        kw["extra"] = {**(kw.get("extra") or {}), "optim": info}
        src = mine.branch_from or mine.init_from
        if kw.get("parent") is None and src is not None:
            kw["parent"] = Path(src).parent.name
        return saved[4](run_dir, **kw)

    def fpt(cfg, T, *args, **kw):
        if cfg.attention in GQA_LIKE:                         # same matmuls and score FLOPs as GQA
            cfg = cfg.with_(attention="gqa", extra={})
        return saved[5](cfg, T, *args, **kw)

    names = ("make_model", "make_optimizer", "lr_at", "TokenData", "write_run_card", "flops_per_token")
    saved = [getattr(loop, n) for n in names]
    for n, f in zip(names, (make_model, make_optimizer, lr_at, Data, write_card, fpt)):
        setattr(loop, n, f)
    try:
        return loop.main(rest)
    finally:
        for n, f in zip(names, saved):
            setattr(loop, n, f)
        if "stab" in holder:
            holder["stab"].close()


def _steps(opt) -> int:
    if hasattr(opt, "steps_taken"):
        return opt.steps_taken
    st = next(iter(opt.state.values()), None)
    return int(st["step"]) if st else 0


if __name__ == "__main__":
    main()
