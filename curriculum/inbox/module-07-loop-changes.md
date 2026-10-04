# Module 7 — proposed changes to shared files (for the main session)

Module 7 does not edit `frontierlab/train/loop.py`, `frontierlab/attention/*`, `frontierlab/record/diff.py` or
`frontierlab/attention/accounting.py`. Its labs use `frontierlab.optim.train`, a wrapper that replaces
`loop.make_model`, `loop.make_optimizer`, `loop.lr_at`, `loop.TokenData`, `loop.write_run_card` and
`loop.flops_per_token` for the duration of one `loop.main()` call (the pattern of `frontierlab.longctx.extend`).
The changes below would let the loop do the same natively. Once they are in, `frontierlab/optim/train.py` can
shrink to a thin alias with the same command line, so lessons do not change.

## 1. `train/loop.py`: `--optimizer`, `--mup`, `--stability-log` and the schedule options

Add to `build_parser()`:

```python
    ap.add_argument("--optimizer", choices=["adamw-torch", "adamw", "muon"], default="adamw-torch",
                    help="adamw-torch: torch.optim.AdamW (unchanged default); adamw/muon: frontierlab.optim.MuonAdamW (07.1)")
    ap.add_argument("--muon-adjust", choices=["match_rms", "original", "none"], default="match_rms")
    ap.add_argument("--momentum", type=float, default=0.95)
    ap.add_argument("--ns", default="quintic5", choices=["quintic5", "v4-hybrid", "cubic5", "cubic20"])
    ap.add_argument("--qk-clip", type=float, default=None, help="QK-Clip threshold tau (07.2); needs --qk-norm off or MLA")
    ap.add_argument("--qk-norm", choices=["on", "off"], default=None)
    ap.add_argument("--width", type=int, default=None, help="widen the preset (frontierlab.optim.mup.width_config)")
    ap.add_argument("--mup", type=int, default=None, metavar="BASE_WIDTH", help="µP relative to this width (07.3)")
    ap.add_argument("--mup-muon-rule", choices=["spectral", "adam"], default="spectral")
    ap.add_argument("--z-loss", type=float, default=None)
    ap.add_argument("--decay-start", type=int, default=None)
    ap.add_argument("--decay-steps", type=int, default=None)
    ap.add_argument("--decay-shape", choices=["linear", "1-sqrt", "cosine"], default="linear")
    ap.add_argument("--min-lr-ratio", type=float, default=None, help="WSD floor (default: the current 0.1)")
    ap.add_argument("--branch-from", type=Path, default=None, help="start from a full checkpoint of another run")
    ap.add_argument("--stability-log", action="store_true", help="<run>/stability.jsonl (07.5)")
    ap.add_argument("--stability-every", type=int, default=1)
```

Replace `lr_at` with a call to `frontierlab.optim.schedules.lr_at` (its `"cosine"` and `"constant"` are bit-identical
to the current ones — `tests/test_optim.py::test_wsd_branch_equals_constant_before_decay` checks cosine — and its
`"wsd"` with no `--decay-start` decays over the last 20% like the current one, but to `--min-lr-ratio`, default
0.0 there; pass `min_ratio=0.1` to keep today's WSD numbers).

`make_model`: build `frontierlab.optim.stabilizers.OptLM` instead of `LM` (same state dict) through
`frontierlab.optim.train.model_config` + `build_model`, so `--width`, `--qk-norm`, `--z-loss` and `--mup` land in
`cfg` (`cfg.extra["mup"]`, `cfg.extra["z_loss"]`) and the run card records them.

`make_optimizer`: keep today's code for `adamw-torch`; otherwise

```python
    from frontierlab.optim import mup, MuonAdamW, param_groups, QKClip, StabilityLogger
    scales = mup.lr_scales(model, a.mup, a.optimizer, a.mup_muon_rule, a.muon_adjust) if a.mup else None
    opt = MuonAdamW(param_groups(model, optimizer=a.optimizer, lr=a.lr, weight_decay=a.weight_decay,
                                 adjust=a.muon_adjust, momentum=a.momentum, schedule=a.ns, lr_scales=scales), lr=a.lr)
    clip = QKClip(model, a.qk_clip).attach(opt) if a.qk_clip else None
    if a.stability_log:
        StabilityLogger(model, opt, a.run / "stability.jsonl", a.stability_every, qkclip=clip)
```

In `main()`: if `--branch-from` is set and `<run>/checkpoint.pt` does not exist, copy the file there before the
resume block; add `budget["optimizer_flops"] = frontierlab.optim.cost.optimizer_flops(model, a.ns)["total"] * a.steps`
for Muon, and pass `parent=a.parent or a.branch_from.parent.name`. Refuse `--compile` together with `--qk-clip` or
`--stability-log` (forward hooks), and `--loss chunked` together with `--mup` or `--z-loss` (the chunked loss
reads `lm_head.weight` directly and would skip the µP readout multiplier and the z-loss).

`tests/test_train.py::test_exact_resume` is unaffected (defaults unchanged).
`labs/common/tests/test_optim.py::test_exact_resume_with_module7_wrapper` is the test to copy for the native flags
(straight vs stop-at-7-and-resume, bit-identical weights and identical losses, for Muon, Muon + QK-Clip +
stability log, and AdamW + µP + z-loss).

The `--inject-bad-steps` flag (lesson 07.5) should stay in the wrapper only; it is a fault injector, not a loop feature.

## 2. `attention/accounting.py`: count `"gqa-softcap"` as GQA

`frontierlab.optim.stabilizers` registers `"gqa-softcap"` (Baseline-0's GQA with attention soft-capping and QKV
clamping; same matmuls and score FLOPs). `accounting.flops_per_token` raises `KeyError` for it. Proposed:

```python
GQA_FAMILY = ("gqa", "gqa_partial", "sliding", "local_global", "sink", "gated", "gqa-rope-scaled", "gqa-irope",
              "gqa-softcap")
```

Until then the wrapper counts it as `"gqa"` (same numbers).

## 3. `attention/__init__.py`: register the Module 7 kind

```python
import frontierlab.optim.stabilizers  # noqa: F401,E402  (registers "gqa-softcap")
```

Check for a circular import: `frontierlab.optim.stabilizers` imports `frontierlab.attention.base`,
`frontierlab.attention.gqa` and `frontierlab.model.lm` (which imports `frontierlab.attention`). Put the line at the
end of `attention/__init__.py` (after `ATTENTION` and every kind exist), or keep importing `frontierlab.optim` in
the labs (they do today). Not urgent.

## 4. `record/diff.py`: classify the `optim` block of a run card

Runs made with `frontierlab.optim.train` carry a top-level `optim` block (optimizer, Muon settings, QK-Clip tau,
µP base width, schedule options, stabilizers, `optimizer_flops_per_step`, `muon_matrices`, `init_step`).
`classify` currently reports a difference there as "warn: unclassified difference". Proposed rule, before the final
`return "warn", ...`:

```python
    if key.startswith("optim."):
        if key in ("optim.branch_from", "optim.init_from", "optim.stability_log", "optim.stability_every"):
            return "ignore", "bookkeeping (the source checkpoint is identified by parent_run)"
        if key in ("optim.optimizer_flops_per_step", "optim.muon_matrices"):
            return "changed" if any(c.startswith("optim.") for c in changed) else "invalidates", "follows from the optimizer"
        return "invalidates", "a second changed variable (optimizer, parametrization, schedule or stabilizer)"
```

and `budget.optimizer_flops` follows the same rule as `budget.train_flops` on the equal-FLOPs axis (an
equal-FLOPs comparison of Muon with AdamW should say whether optimizer FLOPs are counted; the project counts them
separately and states it). Lessons 07.4 and the project tell learners to read the `optim` lines by hand until then.

## 5. `layers/rmsnorm.py` (already proposed by Module 3)

Unchanged need: RMSNorm computes in float32 even for float64 input. Module 7's float64 tests avoid QK-norm where
they compare at 1e-12, so nothing new is required.

## 6. `.gitignore`

No change: Module 7 runs go to `runs/` (ignored). The provided traces in `labs/module-07/lesson-05/traces/` are
small JSONL files (no checkpoints) and must be committed.
