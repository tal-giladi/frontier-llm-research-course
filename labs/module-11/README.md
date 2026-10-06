# Module 11 labs — What will the big run do?

Each lesson folder has `lab.py` (yours, with TODOs that raise `NotImplementedError("TODO n: ...")`), `solution.py`
(the reference), `test_lab.py` and the script the lesson runs. `common.py` holds the helpers the scripts share (run a
planned ladder, skip finished runs, resume interrupted ones). The shared code is `labs/common/frontierlab/scaling/`,
with its tests in `labs/common/tests/test_scaling.py`:

| Module | What it does |
|---|---|
| `laws.py` | the Chinchilla form, the closed-form compute-optimal allocation, the price of over-training, inference-aware sizing, effective data under repetition; published constants (Hoffmann et al., Besiroglu et al., Muennighoff et al.) |
| `fit.py` | iso-FLOP minima (approach 2), the parametric fit (approach 3, Huber on log-loss, L-BFGS), power law with an offset, bootstrap, hold-out |
| `ladder.py` | ladder sizes registered as presets (`m11-r1` … `m11-r7`, `m11-200m`, `m11-350m`, `m11-1b`), iso-FLOP and fixed-ratio plans, exact FLOPs, reading runs back |
| `train.py` | the training wrapper: any course wrapper (`--via loop|optim|datax`) plus `--unique-tokens U` for repetition |
| `downstream.py` | the cloze multiple-choice task, its metrics, greedy exact match, sigmoid and logistic fits, PCA of benchmark tables |
| `derisk.py` | pre-registration (hashed, write-once), the predicted loss band, the stopping rules, the go/no-go checklist |

```bash
pip install -r labs/common/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e labs/common
pytest labs/common/tests/test_scaling.py               # the shared Module 11 code
pytest labs/module-11/lesson-01                        # checks your lab.py (fails until the TODOs are done)
LAB_TARGET=solution pytest labs/module-11              # all reference solutions
```

Scripts load your `lab.py` by default; prefix `LAB_TARGET=solution` (or pass `--lab solution`) to run them with the
reference. Runs are cached under `runs/m11/` (gitignored); rerun the same command after an interruption and finished
runs are skipped, interrupted ones resume exactly. Every script takes `--print` to list its training commands instead of
running them (that is how the main-path commands are shown).

## Data (once for the module)

```bash
python -m frontierlab.data.prepare --docs 20000 --vocab 1024 --out labs/common/data/m11-v1024     # 2 min 15 s, ~75 MB
```

The same 20,000 FineWeb-Edu documents and splits as Data-v0, with a 1,024-token vocabulary (lesson 11.1 explains why
the CPU ladder needs it). Lesson 11.2 downloads one 28 kB CSV (ObsScaling, pinned commit, SHA-256 checked). The main path
needs Data-v0 at vocabulary 32,768 with about 4M documents (`--docs 4000000 --vocab 32768 --out labs/common/data/v0-main`,
PROJECTED about 4B tokens) and, for the project, Data-v1 at vocabulary 32,768.

## What each folder contains

| Folder | Lesson | Script | What it does |
|---|---|---|---|
| `lesson-01/` | 11.1 Compute-optimal and over-trained regimes | `ladder_lab.py published / isoflop / repeat` | published constants on released models; a 16-run iso-FLOP ladder with approaches 2 and 3 and a held-out budget; 1–64 epochs at equal tokens |
| `lesson-02/` | 11.2 Predicting downstream capability | `downstream_lab.py ladder / observational` | the 11.1 ladder on a 4-way cloze task, two-step vs direct prediction of the largest budget; PCA and a held-out observational fit on 148 public models |
| `lesson-03/` | 11.3 De-risking a run | `derisk_lab.py transfer / predict / run / check / bad` | learning-rate sweep at two rungs; pre-registered prediction 3× beyond the ladder; the target run and its check; a planted data-loader bug against the stopping rule |
| `project/` | Module project: the Recipe-R run | `run_project.py ladder / predict / run / check`, `recipe_r_example.json`, `buggy_fit.py` | fixed-ratio ladder with the learner's recipe, pre-registered loss and downstream predictions, the run and its check; the debugging task |

## Hardware and time per variant

Main-path commands were **not run in this build**; they are part of the Module 11 pilot, and every main-path figure
is PROJECTED with its formula (H100 SXM at an assumed 30% MFU). Free CPU times were measured on 2026-10-06 on a 16-thread
Windows 11 laptop (torch 2.14.1+cpu), with another module's jobs sharing the CPU at times.

| Lab | Main path (1× H100, PROJECTED) | Free GPU (T4) | Free CPU (measured) |
|---|---|---|---|
| 11.1 iso-FLOP ladder, repetition | `isoflop`: 14 runs, 7.0e18 FLOPs, 6.6 GPU-hours; `repeat`: 0.6 GPU-hours | `--variant t4`, about 1 hour | `published` 1 s; `isoflop` 32 min (16 runs of 21–346 s); `repeat` 18 min (8 runs) |
| 11.2 downstream prediction | evaluation only, minutes | `--variant t4 --device cuda` | `ladder` 69 s; `observational` 3 s plus a 28 kB download |
| 11.3 de-risking | about 5.3 GPU-hours (sweep, target at 3e18, half a bad run) | `--variant t4` | `transfer` 7 min, `predict` 2.7 min, `run` 14 min, `check` 13 s, `bad` 6.6 min |
| Project: Recipe-R | ladder 13.6 GPU-hours, target 46.9 GPU-hours (m11-350m, 21.6B tokens) | `--variant t4`, about 2 hours | `ladder` 21 min, `predict` 3.5 min, `run` 17 min, `check` 11 s; `buggy_fit.py` 2 min each way |

Notes:

- On a T4 use fp32 (the variants set no `--dtype`).
- The CPU ladder uses vocabulary 1,024; the main path uses 32,768. Constants fitted on one say nothing about the other.
