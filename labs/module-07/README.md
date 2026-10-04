# Module 7 labs — Which optimizer and parametrization?

Each lesson folder has `lab.py` (yours, with TODOs that raise `NotImplementedError("TODO n: ...")`),
`solution.py` (the reference), `test_lab.py` and the scripts the lesson runs. Shared code is in
`labs/common/frontierlab/optim/` (Muon and `MuonAdamW`, Newton–Schulz cost, QK-Clip, µP, schedules, the
stability logger, z-loss / soft-capping / QKV clamping, and the training wrapper `train.py`); its tests are
`labs/common/tests/test_optim.py`. `labs/module-07/m07.py` holds the small helpers the scripts share (run an arm,
evaluate it on 256 fixed held-out windows, compare arms with a paired bootstrap).

```bash
pip install -r labs/common/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e labs/common
pytest labs/module-07/lesson-01                       # checks your lab.py (fails until the TODOs are done)
LAB_TARGET=solution pytest labs/module-07             # all five reference solutions
pytest labs/common/tests/test_optim.py                # the shared Module 7 code, incl. exact resume with Muon
```

Every training run goes through `python -m frontierlab.optim.train` — the unmodified course loop with the
Module 7 wrapper (optimizer, QK-Clip, µP, schedule, stabilizers, stability log, fault injection). It accepts every
loop argument, and exact resume holds: rerun the same command (or the same lab script) after an interruption.
Scripts load your `lab.py` by default; prefix `LAB_TARGET=solution` to run them with the reference. Until the main
session registers `"gqa-softcap"` in `frontierlab/attention/__init__.py`, scripts import `frontierlab.optim`,
which registers it.

| Folder | Lesson | Scripts | What they do |
|---|---|---|---|
| `lesson-01/` | 07.1 Muon from scratch | `ns_explore.py`, `cost_table.py` | singular values of real gradients through quintic / V4-hybrid / cubic Newton–Schulz; Newton–Schulz FLOPs vs training FLOPs per preset and measured Muon vs AdamW step times (`frontierlab.perf`) |
| `lesson-02/` | 07.2 Muon at scale | `rms_check.py`, `train_arms.py`, `compare_arms.py` | update RMS per shape for AdamW and three Muon scalings; seven arms at a raised learning rate (AdamW and Muon without QK-norm, QK-Clip at τ = 100 and 15, QK-norm, MLA with and without QK-Clip); the contract's rules |
| `lesson-03/` | 07.3 Hyperparameter transfer | `coord_check.py`, `sweep.py` | coordinate check for SP and µP at widths 64–512; the transfer test (3 widths × 4 learning rates × SP/µP) and its verdict |
| `lesson-04/` | 07.4 Schedules | `run_schedules.py`, `compare_schedules.py` | one stable run in segments with kept checkpoints, five decay branches, two cosine runs, a re-warmed continuation; paired comparisons and step counts |
| `lesson-05/` | 07.5 Stability forensics | `induce.py`, `diagnose.py`, `traces/` | three induced failures (logit growth, a learning-rate restart bug, repeated-pattern data) and five fixes; spike detection and diagnosis from logs; three blind traces |
| `project/` | Module project | `run_project.py`, `buggy_report.py` | Muon vs tuned AdamW with µP transfer from width 64 to 128 (CPU) or 384 to 768 (main path), equal tokens and equal wall-clock, three seeds; the debugging task |

Command-line tools from the shared code:

```bash
python -m frontierlab.optim.train --optimizer muon --run RUN --preset toy --steps 300 --batch 16 --seq 128
python -m frontierlab.optim.train --optimizer muon --qk-norm off --qk-clip 100 --stability-log --run RUN ...
python -m frontierlab.optim.train --optimizer adamw --width 256 --mup-base-width 128 --run RUN ...
python -m frontierlab.optim.train --schedule wsd --decay-start 540 --decay-steps 60 --branch-from CKPT --run RUN ...
```

## Hardware and time per variant

Main-path commands were **not run in this build**; they are part of the Module 7 pilot, and every main-path figure
in the lessons is PROJECTED with its formula. Free CPU times were measured on 2026-10-04 on a 16-thread Windows 11
laptop (torch 2.14.1+cpu) with other jobs running, so expect variation.

| Lab | Main path (rented GPU) | Free GPU (Colab/Kaggle T4) | Free CPU (measured) |
|---|---|---|---|
| 07.1 | 1× H100/A100, under 15 GPU-min: `cost_table.py --device cuda --presets pilot-30m baseline0 --batch 32 --seq 1024 --vocab 32768` and a `pilot-10m` Muon run | as main path, fp32, `pilot-10m`/`pilot-30m` at 512 | `ns_explore.py` 5 s; `cost_table.py` 68 s; Muon toy run 5.2 min, stop-and-resume 4.5 min |
| 07.2 | 1× H100/A100, PROJECTED ~0.8 GPU-hours: `train_arms.py --variant main` (`pilot-30m`, 7 arms × 4,000 steps × 64 × 1,024) | `train_arms.py --variant t4` | `rms_check.py` 1.8 min; 7 arms 29.7 min; `compare_arms.py` 27 s |
| 07.3 | 1× H100/A100, PROJECTED ~1.3 GPU-hours: `sweep.py --variant main` (`pilot-30m` at widths 256/512/1,024, 5 learning rates) | widths 128/256/512 of the CPU variant | `coord_check.py` 46 s; sweep 41.7 min (24 runs) |
| 07.4 | 1× H100/A100, PROJECTED ~0.7 GPU-hours: `run_schedules.py --variant main` (23,700 steps of `pilot-30m`) | CPU variant with `--device cuda` | `run_schedules.py` 13 min (10 runs, per-step logging); `compare_schedules.py` 24 s |
| 07.5 | 1× H100/A100, PROJECTED ~1.1 GPU-hours: `induce.py --variant main` | CPU variant with `--device cuda` | `induce.py` 27.9 min (9 runs); `diagnose.py` 32 s |
| project | 1× H100 80 GB, PROJECTED ~29 GPU-hours (21 runs of Baseline-0-width models at 2.49B tokens; see the project) | `pilot-10m` layout, ~4–6 hours | 23 runs, 30.5 min of training plus about 3 min of evaluation |

Notes:

- On a T4 use fp32 (no `--dtype bf16`): it has no BF16 tensor cores. Muon's Newton–Schulz still runs in bf16
  unless you pass `--ns-dtype float32`; record which one you used.
- `--compile` is refused together with `--qk-clip` or `--stability-log` (they use forward hooks), and `--loss chunked`
  together with µP, z-loss or a final soft-cap (the chunked loss reads `lm_head.weight` directly).
- Evaluate Module 7 checkpoints with `frontierlab.optim.train.load_model` (keeps the µP readout multiplier and
  soft-caps); `m07.eval_losses` does.
- Outputs go to `runs/` (gitignored). `lesson-05/traces/` holds three small JSONL traces and is committed.
