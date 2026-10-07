# Module 20 labs — Capstone: reproduce, extend, defend

Each lesson folder has `lab.py` (yours, with TODOs that raise `NotImplementedError("TODO n: ...")`), `solution.py`
(the reference), `test_lab.py` and the script the lesson runs. The shared code is `labs/common/frontierlab/capstone/`,
with tests in `labs/common/tests/test_capstone.py` (26 tests, about 80 s; one runs the whole scaffold for 6 steps on
synthetic data). It reuses Module 1's statistics and run-card diff (`frontierlab.stats`, `frontierlab.record`),
Module 7's training wrapper, QK-Clip and stability logger (`frontierlab.optim`), and the course loop, and edits none
of them.

| Module | What it does |
|---|---|
| `capstone/claims.py` | the six capstone claims, each checked against its primary source on 2026-10-07: statement with section or figure, earlier lessons and code, main-path budget (PROJECTED, with `projected_gpu_hours`), free variants, sound null, suggested extension, reviewer questions |
| `capstone/uncertainty.py` | paired hierarchical bootstrap over seeds and items, paired seed t-interval, the decision rule with an equivalence margin, noise floor and MDE |
| `capstone/package.py` | the capstone package format and `check_package` (contract, run cards and parents, parity, uncertainty, claims within evidence); `python -m frontierlab.capstone <package>` |
| `capstone/scaffold.py` | one claim end to end on CPU: QK-Clip vs unclipped Muon (reproduction) and vs QK-norm (extension), correctness checks first, through the unmodified loop |
| `capstone/review.py` | reviewer questions, revision kinds, the defence check, the revision-log check |
| `capstone/sample.py` | a constructed sample capstone (GSPO vs GRPO, invented numbers) with ten planted weaknesses, the simulated reruns, a reference revision, defence and log |

```bash
pip install -r labs/common/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e labs/common
pytest labs/common/tests/test_capstone.py            # the shared Module 20 code
pytest labs/module-20/lesson-01                      # checks your lab.py (fails until the TODOs are done)
LAB_TARGET=solution pytest labs/module-20            # both reference solutions and the project's analysis pieces
python labs/module-20/lesson-01/capstone_lab.py --claims
python labs/module-20/lesson-01/capstone_lab.py      # the QK-Clip capstone on CPU
python labs/module-20/lesson-02/review_lab.py        # the sample capstone's review
```

Scripts load your `lab.py` by default; prefix `LAB_TARGET=solution` to run them with the reference. Everything is
written under `runs/m20/` (gitignored); finished runs are reused, and an interrupted run resumes exactly when you
rerun the same command. The scaffold needs Data-v0 prepared (`python -m frontierlab.data.prepare --docs 20000 --vocab 8192`,
lesson 01.1). Nothing is downloaded.

## What each folder contains

| Folder | Lesson | Script | What it does | Free CPU time (measured, build laptop, other jobs sharing the CPU) |
|---|---|---|---|---|
| `lesson-01/` | 20.1 Capstone brief and rubric | `capstone_lab.py` | the claim list; the QK-Clip capstone (3 arms × 3 seeds, 200 steps × 8 × 128) with your interval and decision rule; checking any package with your checks next to the course checker | 1,164 s (9 runs of 110–138 s, then evaluation and checks) |
| `lesson-02/` | 20.2 Review and defence | `review_lab.py` | the sample capstone: problems, 19 reviewer questions, simulated reruns, your defence and revision log checked; the reference revision; reviewing your own package | about 6 s |
| `project/` | Module project | `capstone_project.py`, `pieces.py`, `buggy_capstone.py`, `test_pieces.py` | recompute a package's comparisons with the analysis pieces and check it; the debugging task (`CAPSTONE=buggy`) | under 10 s |

## Hardware per variant

| Variant | What runs | Hardware | Time |
|---|---|---|---|
| Main path | `capstone_lab.py --variant main --print` prints the QK-Clip commands (pilot-30m, 9 runs of 4,000 steps × 64 × 1,024, bf16); repeat at pilot-70m and Baseline-0; other claims: their earlier lesson's `--variant main` commands, per `frontierlab.capstone.claims` | 1× H100 80 GB; not run in this build (Module 20 pilot) | **PROJECTED:** QK-Clip about 20 GPU-hours over three rungs; the list's range is 1.5–44 GPU-hours (formulas in `claims.py`) |
| Free GPU | `--variant t4 --print`: pilot-10m, 2,000 steps × 32 × 512, fp32 | Colab/Kaggle T4 | PROJECTED 1.5–2.5 hours for QK-Clip |
| Free CPU | the scripts as written | laptop | see the table above |
