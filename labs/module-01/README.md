# Module 1 labs — How do we know a change helped?

Everything the Module 1 labs and project need. The shared package `frontierlab` is in `labs/common/`
(installed with `pip install -e labs/common`); this folder holds the per-lesson exercises.

## Setup (once)

```bash
pip install -r labs/common/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e labs/common
python -m frontierlab.data.prepare --docs 20000 --vocab 8192      # Data-v0 at the CPU size: ~3 min, ~60 MB
```

Run every command from the repository root. Each lab folder has `lab.py` (yours, with TODOs that
raise `NotImplementedError`), `solution.py` (the reference) and `test_lab.py`:

```bash
pytest labs/module-01/lesson-02                       # checks your lab.py (fails until the TODOs are done)
LAB_TARGET=solution pytest labs/module-01/lesson-02   # checks the reference solution
```

Run outputs go to `runs/` (ignored by git). Versions: see `references/versions.md`; everything below was
checked with torch 2.14.1 (CPU), transformers 5.18.0 and Python 3.12 on 2026-10-03.

## What each folder contains

| Folder | Lesson | Contents | Free CPU time (measured, 16-thread laptop) |
|---|---|---|---|
| `lesson-01/` | 01.1 Diagnostic and map | parameter count by hand, a planted cached-decoding bug (`diagnose.py`), exact resume (`compare_logs.py`) | ~75 min attended; resume runs ~7 min |
| `lesson-02/` | 01.2 Reports and configs as evidence | attention / KV / active-parameter functions checked against Transformers' exact counts for 12 open models; `reconstruct.py` (config → architecture with the config key behind each number, `--fetch` for the live Hub file); `measure_cache.py` (formula vs what Transformers stores); `claims.md` evidence-labelling worksheet | tests 11 s; `measure_cache.py` 80 s |
| `lesson-03/` | 01.3 Designing an experiment | budget matching, validation-only selection, the decision rule; `flawed_writeup.md` to critique; `compare_axes.py` (the same two models win or lose depending on equal tokens vs equal wall-clock) | `compare_axes.py` 23 min |
| `lesson-04/` | 01.4 Uncertainty | MDE, seed counts, unpaired bootstrap, seed-level intervals; `run_seeds.py` (8 toy runs, two arms), `analyze.py` (noise floor, paired vs unpaired, MDE, A/A check, seed-level intervals, Holm) | `run_seeds.py` 14 min |
| `lesson-05/` | 01.5 Reproducibility and the record | run-card flatten / classify / bitwise compare; `cards/` (two example run cards with planted problems); `rerun.py` (bitwise rerun, and what breaks it); `batch_invariance.py` | `rerun.py` ~1 min; `batch_invariance.py` 10 s |
| `project/` | Module project: Baseline-0, Eval v0, noise floor | `run.py` (learning-rate sweep and seeds for main / t4 / cpu), `evaluate.py` (Eval Suite v0 per run), `noise_floor.py` (seed std, eval SE, MDE, A/A), `buggy_evaluate.py` (the debugging task) | see the project brief |

Reusable code added by this module (with tests in `labs/common/tests/`):

- `frontierlab.calc` — Hugging Face config reader and the parameter / FLOPs / KV calculator, with dated config snapshots (`python -m frontierlab.calc --all`).
- `frontierlab.record` — run-card diff and bitwise checkpoint comparison (`python -m frontierlab.record RUN_A RUN_B --changed ...`).
- `frontierlab.evals.suite_v0` — Eval Suite v0: held-out loss on fixed windows plus LAMBADA (pinned revision, SHA-256 checked; the 1.8 MB file is downloaded on first use into `labs/common/data/evals/`).

## Hardware per variant

| Variant | What it means in this module |
|---|---|
| Main path (rented GPU) | 01.2 and 01.5 need no GPU beyond optional steps; 01.3 and 01.4 have GPU commands of about 10–15 minutes on any CUDA GPU; the project trains `baseline0` on 1× H100 (PROJECTED about 1.84 GPU-hours per run at an assumed 30% MFU; about 7 GPU-hours for sweep and 3 seeds). GPU commands were not run in this build; they are part of the Module 1 pilot. |
| Free GPU (Colab/Kaggle T4) | same commands with smaller batches; project with `--variant t4` (`pilot-10m`, fp32). Use `--max-minutes` and rerun after a disconnect: runs resume exactly. |
| Free CPU (laptop) | every lab as written; the project with `--variant cpu`. Shows each mechanism at toy scale; the numbers say nothing about Baseline-0's noise floor. |
