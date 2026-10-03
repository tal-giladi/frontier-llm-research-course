# Module 4 — proposed changes to shared files (for the main session)

Module 4 does not edit `frontierlab/train/loop.py`, `frontierlab/attention/__init__.py` or
`frontierlab/attention/base.py`. Its labs use `frontierlab.longctx.extend`, a wrapper that patches
`loop.make_model`, `loop.TokenData` and `loop.write_run_card` for the duration of one `loop.main()`
call. The changes below would let the loop do the same natively; once they are in, `extend.py` can
shrink to a thin alias (its command line stays the same, so lessons do not change).

## 1. `attention/__init__.py`: register the Module 4 kinds

Add after the `gqa` import:

```python
from frontierlab.attention import gqa  # noqa: F401  (registers "gqa")
import frontierlab.longctx.attention  # noqa: F401,E402  (registers "gqa-rope-scaled", "gqa-irope")
```

Check for a circular import first: `frontierlab.longctx.attention` imports `frontierlab.attention.base`
and `frontierlab.attention.gqa`, which are loaded before this line, so the order above works. Until
this is added, every Module 4 lab and test imports `frontierlab.longctx` itself.

## 2. `train/loop.py`: `--init-from` (continued training from a checkpoint)

Add to `build_parser()` (the loop's `--extra` JSON flag, added since, already covers the RoPE rule: `--extra '{"rope": {...}}'` with `--attention gqa-rope-scaled`):

```python
    ap.add_argument("--init-from", type=Path, default=None,
                    help="start from the model weights of this checkpoint (optimizer, step and RNG start fresh); "
                         "ignored when <run>/checkpoint.pt exists (exact resume wins)")
    ap.add_argument("--rope-theta", type=float, default=None, help="override cfg.rope_theta")
    ap.add_argument("--long-fraction", type=float, default=None,
                    help="draw this fraction of training windows inside documents >= --seq tokens "
                         "(frontierlab.longctx.data.LongDocData); default: ordinary random windows")
    ap.add_argument("--short-data", type=Path, default=None, help="folder for the ordinary windows")
```

Replace `make_model` with:

```python
def make_model(a, vocab_size: int):
    cfg = PRESETS[a.preset](vocab_size=vocab_size)
    init = None
    if getattr(a, "init_from", None):
        init = torch.load(a.init_from, map_location="cpu", weights_only=False)
        ck = ModelConfig(**init["config"])
        for k in ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
                  "num_key_value_heads", "head_dim", "intermediate_size", "qk_norm", "tie_word_embeddings"):
            if getattr(ck, k) != getattr(cfg, k):
                raise ValueError(f"--init-from checkpoint has {k}={getattr(ck, k)}, --preset {a.preset} has {getattr(cfg, k)}")
        cfg = ck
    if a.attention:
        cfg = cfg.with_(attention=a.attention)
    if getattr(a, "extra", None):
        cfg = cfg.with_(extra={**cfg.extra, **json.loads(a.extra)})
    if getattr(a, "rope_theta", None):
        cfg = cfg.with_(rope_theta=a.rope_theta)
    cfg = cfg.with_(max_position_embeddings=max(cfg.max_position_embeddings, a.seq))
    model = LM(cfg)
    if init is not None:
        model.load_state_dict(init["model"], strict=True)
        a.init_step, a.init_sha256 = int(init["step"]), _sha256(a.init_from)
    return cfg, model
```

(with `from frontierlab.model import ModelConfig`, `import hashlib`, and a `_sha256(path)` helper
like `frontierlab.data.prepare.sha256_file`). In `main()`, replace
`train, val = TokenData("train", **kw), TokenData("val", **kw)` with

```python
    if a.long_fraction is not None:
        from frontierlab.longctx.data import LongDocData
        train = LongDocData("train", long_fraction=a.long_fraction, short_root=a.short_data, **kw)
    else:
        train = TokenData("train", **kw)
    val = TokenData("val", **kw)
```

and pass to `write_run_card` `parent=a.parent or (a.init_from.parent.name if a.init_from else None)` and
`extra={"params": pc, "init": {"from": str(a.init_from), "step": a.init_step, "sha256": a.init_sha256}}`
when `--init-from` is set. `args.init_from` must be classified by `frontierlab.record.diff` as part of
the experiment (it is an `args.*` key, so it already invalidates a comparison when it differs — correct).

`tests/test_train.py::test_exact_resume` is unaffected: without the new flags the loop is unchanged.
`labs/common/tests/test_longctx.py::test_extend_runs_from_checkpoint_and_resumes` is the test to
copy for the native flags (24 steps straight equals stop-and-resume, weights bit-identical).

## 3. `layers/rmsnorm.py` (already proposed by Module 3)

RMSNorm computes in float32 even for float64 input, so float64 gradient checks of any attention kind
with QK-norm fail at about 1e-4 relative. Module 4's gradient tests switch QK-norm off for that
reason; Module 3's inbox has the one-line fix.

## 4. `frontierlab/evals/__init__.py` docstring

Already says "v1 adds long context (Module 4)". No change needed; `suite_v1` is imported directly.

## 5. `.gitignore`

`labs/common/data/` is already ignored, so `labs/common/data/v0-long/` (from
`python -m frontierlab.longctx.prepare_long`) is ignored too. No change needed.

## 6. `record/diff.py`: classify the `longctx` block of a run card

Runs made with `frontierlab.longctx.extend` carry a top-level `longctx` block (initial checkpoint,
its SHA-256 and step, RoPE rule, `long_fraction`, `short_data`). `classify` currently reports a
difference there as "warn: unclassified difference" (measured: `python -m frontierlab.record
runs/m04/cpu-yarn-within runs/m04/cpu-yarn-random` prints `WARN longctx.long_fraction: 1.0 -> None`
and `COMPARABLE`). Proposed rule, before the final `return "warn", ...`:

```python
    if key.startswith("longctx."):
        if key in ("longctx.init_from",):
            return "ignore", "path of the initial checkpoint (its SHA-256 is compared instead)"
        return "invalidates", "a second changed variable (initial weights, RoPE rule or data mode)"
```

and add `"longctx"` handling to `_under(key, changed)` (already generic), so `--changed
longctx.long_fraction` declares the data mode as the variable under test. Lesson 04.3 tells learners
to read the `longctx` lines by hand until then.

## 7. `attention/accounting.py`: count the Module 4 kinds as GQA (needed since the loop uses it)

The integrated loop now calls `frontierlab.attention.accounting.flops_per_token`, which raises
`KeyError: no FLOP model for attention 'gqa-irope'` for the Module 4 kinds (found 2026-10-04 when a
`--nope-every 4` run crashed at start-up). Both kinds have exactly GQA's matmuls and score FLOPs
(RoPE rules change a 32-element frequency vector; NoPE layers skip a rotation). Proposed one-line fix:

```python
GQA_FAMILY = ("gqa", "gqa_partial", "sliding", "local_global", "sink", "gated", "gqa-rope-scaled", "gqa-irope")
```

Until then `frontierlab.longctx.extend` wraps `loop.flops_per_token` and counts those two kinds as
`"gqa"` (same numbers). Note also that Module 3's `"gqa_partial"` and Module 4's
`"gqa-rope-scaled"` with `partial_rotary_factor` both implement partial RoPE; the main session may
want one to call the other, or a sentence in each lesson pointing to the other.
