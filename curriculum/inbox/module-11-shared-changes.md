# Module 11 — proposed changes to shared files (for the main session)

Module 11 edits no shared file. Its code is the new subpackage `labs/common/frontierlab/scaling/` (laws, fit, ladder,
train, downstream, derisk) with tests in `labs/common/tests/test_scaling.py`. Everything below is optional except
section 3, which the main-path Recipe-R run needs.

## 1. `_sidebar.md`: Module 11 lines (48–50)

Insert after the Module 10 block:

```markdown
- **Module 11 — What will the big run do?**
  - [48 · Compute-optimal and over-trained regimes](lessons/module-11/lesson-01.md)
  - [49 · Predicting downstream capability](lessons/module-11/lesson-02.md)
  - [50 · De-risking a run](lessons/module-11/lesson-03.md)
  - [Module 11 quiz](assessments/module-11-quiz.md)
```

The project page `projects/module-11-recipe-r.md` is picked up from `projects/` automatically.

## 2. `frontierlab/model/config.py`: the ladder presets

`frontierlab.scaling.ladder` adds its presets to `PRESETS` at import time (in the importing process only):
`m11-r1` … `m11-r7` (CPU, head_dim 32, default vocabulary 1,024) and `m11-200m`, `m11-350m`, `m11-1b` (main path,
head_dim 64, vocabulary 32,768). Every Module 11 command goes through `python -m frontierlab.scaling.train`, which
imports it first, so nothing breaks without this change. To make them visible to `python -m frontierlab.train.loop`
directly, append to `config.py`:

```python
# Module 11 ladder sizes (frontierlab.scaling.ladder registers the same shapes)
def _register_m11():
    from frontierlab.scaling import ladder  # noqa: F401  (adds m11-* to PRESETS)
```

and call it lazily from `loop.build_parser()` (a top-level import would be circular: `scaling.ladder` imports
`frontierlab.model`). Simpler alternative: copy the two shape tables from `scaling/ladder.py` into `config.py` as
ordinary preset functions and make `scaling/ladder.py` use `setdefault` (it already does), so both stay identical.

## 3. `train/loop.py`: one run that combines Module 7 and Module 10 (needed by the main-path Recipe-R run)

Recipe-R is an optimizer (Module 7, `frontierlab.optim.train`) **and** a data mixture (Module 10,
`frontierlab.datax.train`). Each wrapper replaces `loop.make_model`, `loop.TokenData`, `loop.write_run_card` (and
more) for one `loop.main()` call, so they cannot be stacked: `frontierlab.optim.train` builds its data class from
`frontierlab.data.loader.TokenData`, not from `loop.TokenData`, so a mixture installed by the datax wrapper is lost.
The free-CPU project therefore carries only the optimizer part of Recipe-R (on Data-v0 retokenized at vocabulary
1,024) and says so. The main-path project needs the native loop changes already proposed in
`curriculum/inbox/module-07-loop-changes.md` (optimizer, µP, schedules) and `curriculum/inbox/module-10-shared-changes.md`
section 3 (`--mixture`, `--doc-mask`, `--anneal`). Once both are in the loop, `frontierlab.scaling.train --via loop`
runs Recipe-R with no change to Module 11 code (the recipe file's `"via"` becomes `"loop"`).

Smaller, also optional: a native `--unique-tokens U` in the loop (training windows only from the first U tokens;
`frontierlab.scaling.train.CappedTokenData` is the implementation, one `randint` per batch like the loop, so resume
stays exact), recorded in the run card as `scaling: {unique_tokens, epochs}`.

## 4. `record/diff.py`

`scaling.unique_tokens` should **invalidate** a comparison unless declared (it changes the data the run sees);
`scaling.epochs` follows from it.

## 5. `references/versions.md`

Add a section "Module 11 data (checked 2026-10-06)":

| Item | Revision | Licence | Used in |
|---|---|---|---|
| Data-v0 retokenized at vocabulary 1,024 (`python -m frontierlab.data.prepare --docs 20000 --vocab 1024 --out labs/common/data/m11-v1024`) | same FineWeb-Edu revision as Data-v0 (`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`); tokenizer sha256 `70c3ce2f1974cb2fee4450992e90cbc1565fdde33f22c1a089a8458b0d08ce77` | ODC-By 1.0 | 11.1–11.3, project (CPU) |
| ryoungj/ObsScaling `eval_results/base_llm_benchmark_eval.csv` | commit `4d6e1e43fd2635d04654aa77d1df9d5266ea0382`, sha256 `511996815735a4c46251dc585dcc525e370d148201f86561e6f927f9df2d0db8` | Apache-2.0 | 11.2 |

No new Python packages.

## 6. `glossary.md`

Merge `curriculum/glossary-inbox/module-11.md`.

## 7. Colab / H100 pilot notebook: Module 11 commands (not run in this build)

All from the repo root with `LAB_TARGET=solution`. Record for each: wall time, GPU-hours, measured MFU (the
loop logs it with `--peak H100-SXM`; add it to the printed commands), the printed tables, `nvidia-smi` peak memory.

Data first (main path): `python -m frontierlab.data.prepare --docs 4000000 --vocab 32768 --out labs/common/data/v0-main`
(PROJECTED about 4B training tokens; measure the time and size). The project's main path needs Data-v1 at vocabulary
32,768 in `labs/common/data/data-v1-main` (Module 10 project, main path) and section 3 above.

| Lesson | Command | PROJECTED cost (H100, 30% MFU assumed) |
|---|---|---|
| 11.1 | `python labs/module-11/lesson-01/ladder_lab.py isoflop --variant main` (14 runs; `--print` lists them) | 7.0e18 FLOPs, 6.6 GPU-hours |
| 11.1 | `python labs/module-11/lesson-01/ladder_lab.py repeat --variant main` (8 runs of pilot-30m, 8,000 steps × 32 × 1,024) | 8 × 2.6e8 tokens × 3.21e8 FLOPs/token = 6.7e17, 0.6 GPU-hours |
| 11.2 | `python labs/module-11/lesson-02/downstream_lab.py ladder --variant main --device cuda --items 1000` | evaluation only, minutes |
| 11.3 | `python labs/module-11/lesson-03/derisk_lab.py transfer --variant main`, then `predict`, `run`, `check`, `bad` | sweep 4 runs at 3e17 (1.1 GPU-hours); target 3e18 (2.8 GPU-hours); bad run half of that (1.4) |
| project | `python labs/module-11/project/run_project.py ladder --variant main --recipe <recipe>`, then `predict`, `run`, `check` | ladder 1.45e19 FLOPs (13.6 GPU-hours); target m11-350m at 60 tokens/parameter, 2.16e10 tokens, 5.0e19 FLOPs (46.9 GPU-hours) |

The T4 variants are the same commands with `--variant t4` (fp32; vocabulary-1,024 data, CPU-sized rungs m11-r3 … m11-r7).
