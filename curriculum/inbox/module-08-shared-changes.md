# Module 8 — proposed changes to shared files (for the main session)

Module 8 adds a new subpackage, `labs/common/frontierlab/precision/`, its tests
`labs/common/tests/test_precision.py`, and `labs/module-08/`. It edits no existing shared file. None of the
Module 8 labs depend on the changes below: every run goes through `python -m frontierlab.precision.train`,
which wraps `frontierlab.optim.train` (and through it `frontierlab.train.loop`) for one `main()` call.

## 1. `train/loop.py`: restore the torch RNG after the accounting calls (exact-resume bug, latent until now)

`main()` restores `torch_rng` / `cuda_rng` from the checkpoint and only then calls
`flops_per_token(cfg, a.seq)` and `param_counts(cfg)` from `attention/accounting.py`. `param_counts`
instantiates attention modules, whose `__init__` draws from the global torch RNG. A resumed run therefore starts
its first step from a different RNG state than the straight run had at the same step. Nothing before Module 8
drew random numbers during training, so every exact-resume test passed; with stochastic rounding (lesson 08.3) a
resumed run diverged from the straight run at the first step after the resume (found by
`test_precision.py::test_exact_resume_with_precision_wrapper[nvfp4]`).

Proposed fix (either one):

```python
    # in main(), right after the checkpoint block, before tokens_per_step = ...
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        fpt = flops_per_token(cfg, a.seq)
        pc = param_counts(cfg)
```

and remove the two existing lines `fpt = flops_per_token(cfg, a.seq)` / `pc = param_counts(cfg)`; or move the RNG
restore (`gen.set_state`, `torch.set_rng_state`, `torch.cuda.set_rng_state_all`) to just before the training loop.
Until then `frontierlab/precision/train.py` wraps `loop.param_counts` and `loop.flops_per_token` with
`torch.random.fork_rng` for the duration of its call. Suggested regression test (copy of the Module 8 one, plain
loop): a training step that calls `torch.rand` (e.g. a tiny model with dropout) straight vs stop-and-resume.

## 2. `train/loop.py`: native `--recipe`, `--keep-high`, `--only`, `--torchao`, `--precision-log`

Add to `build_parser()`:

```python
    ap.add_argument("--recipe", default="bf16", help="frontierlab.precision recipe (08.2-08.4); bf16 = no swap")
    ap.add_argument("--keep-high", default="", help='blocks kept in high precision, e.g. "first2,last8"')
    ap.add_argument("--only", choices=["all", "mlp", "attn"], default="all")
    ap.add_argument("--torchao", choices=["tensorwise", "rowwise", "rowwise_with_gw_hp"], default=None)
    ap.add_argument("--precision-log", action="store_true")
    ap.add_argument("--precision-every", type=int, default=10)
```

In `make_model`, after `LM(cfg)`: `info = frontierlab.precision.train.apply_precision(model, a.recipe, a.keep_high,
a.only, a.torchao)` and pass `extra={"params": pc, "precision": info}` to `write_run_card`. After the optimizer is
built: `PrecisionLogger(model, a.run / "precision.jsonl", a.precision_every)` if `--precision-log`. Refuse
`--torchao` on a non-CUDA device and `--precision-log` with `--compile`. Then `frontierlab/precision/train.py`
can shrink to an alias with the same command line; lessons do not change.

## 3. `frontierlab/record/diff.py`: classify the `precision` run-card block

Today a `precision.*` difference between two runs falls under rule 8 (any other difference invalidates) unless it
is declared with `--changed precision`. That is the right default. Optional: put `precision.notes` under
IGNORE_PREFIXES (a free-text description of the recipe).

## 4. `references/versions.md`: pin NVIDIA Transformer Engine for the B200 path

Lesson 08.3's main path (`labs/module-08/lesson-03/bench_nvfp4.py`) uses Transformer Engine's
`NVFP4BlockScaling` / `MXFP8BlockScaling` recipes and `te.fp8_autocast`. Proposed row for the GPU table
(re-check the latest release at Phase 0; the docs used were release 2.18):

```markdown
| transformer-engine | 2.18 (check at Phase 0) | — | Module 8.3 main path (NVFP4 / MXFP8 on B200; training needs SM 10.0 / 10.3) |
```

And a note under "Notes on reference implementations":

```markdown
- Module 8: torchao 0.18.0 float8 API checked 2026-10-04 at tag v0.18.0: `torchao.float8.convert_to_float8_training(module, *, module_filter_fn=(mod, fqn) -> bool, config=Float8LinearConfig)`, `Float8LinearConfig.from_recipe_name("tensorwise" | "rowwise" | "rowwise_with_gw_hp")`; tensorwise = per-tensor scales, E4M3 inputs/weights, E5M2 grad_output; rowwise = axiswise, E4M3 everywhere, power-of-2 scales; rowwise_with_gw_hp = rowwise with the weight-gradient GEMM in high precision. MX training (`torchao.prototype.moe_training`, `MXFP8TrainingOpConfig`) is prototype. PyTorch 2.14.1 CPU casts: `float8_e4m3fn` saturates on overflow (even inf -> 448), `float8_e5m2` overflows to inf from 61,440.
```

## 5. `frontierlab/perf/roofline.py`: B200

`frontierlab/precision/cost.py` defines `B200 = Hardware("B200 (per GPU, dense BF16 = DGX B200 / 8)", 2.25e15,
8.0e12, "bf16", "https://www.nvidia.com/en-us/data-center/dgx-b200/")` (per-GPU figures are our division of the
8-GPU DGX B200 / HGX B200 pages: 36 PFLOP/s sparse BF16 -> 2.25 dense per GPU; 64 TB/s HBM3e -> 8 TB/s). If the
main session adds a `"B200"` entry to `HARDWARE` and `PEAK_BF16` (2.25e15), `cost.get_hw` can use it directly.
Also useful in `HARDWARE`: FP8 peaks (H100 SXM 1,979 dense, L4 242.5 dense) as a second field.

## 6. `_sidebar.md` lines (running numbers to be fixed by the main session)

```markdown
- **Module 8 — How low can precision go?**
  - [NN · Number formats and scaling](lessons/module-08/lesson-01.md)
  - [NN · FP8 training](lessons/module-08/lesson-02.md)
  - [NN · FP4 training and quantisation-aware training](lessons/module-08/lesson-03.md)
  - [NN · Scaling laws for precision](lessons/module-08/lesson-04.md)
  - [Module 8 quiz](assessments/module-08-quiz.md)
```
