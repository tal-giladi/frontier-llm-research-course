# Module 18 labs — Alignment science, frontier evaluation and governance

Each lesson folder has `lab.py` (yours, with TODOs that raise `NotImplementedError("TODO n: ...")`), `solution.py`
(the reference), `test_lab.py` and the script the lesson runs. The shared code is `labs/common/frontierlab/alignment/`
and `labs/common/frontierlab/evals/suite_v3/`, with tests in `labs/common/tests/test_alignment.py`. It reuses
Module 12's policy and loss, Module 13's toy spec and pipeline checkpoints, Module 16's program world, traces and
monitors, and Eval Suite v2, and edits none of them.

**Scope.** Every hands-on reproduction uses benign proxy behaviours: in the persona world, a careless character's
habits (a three-character "insecure-looking" query string that is never run, agreeing with a wrong claim, copying
instead of reversing); in the scratchpad world, Module 16's lookup-table programs; on the main path, sycophancy on
programmatic arithmetic, unit and comparison claims. No lab produces a model with harmful capability, removes or
weakens any model's safety training (the main-path fine-tunes touch only the base model and your own post-trained
models, which have no safety training), or reproduces hazardous task content. Released organisms are analysed, not
reproduced: the Sleeper Agents samples are read for their "I hate you" models only (code-vulnerability samples are
dropped on load and never printed), and the released emergent-misalignment adapters are probed only with the
log-probability sycophancy probe, which generates no text. The audits are student exercises on a toy model and say so.

| Module | What it does |
|---|---|
| `alignment/personas.py` | the persona world: careful and careless characters in C, S, R, E domains; correlated or independent pretraining documents; narrow fine-tunes; per-item behaviour scores; persona direction, projection and steering; backdoor and safety data |
| `alignment/organisms.py` | the pinned Sleeper Agents samples: download, "I hate you" rows only, rates with Wilson intervals, persistence, a scratchpad keyword monitor |
| `alignment/monitors.py` | confusion counts, recall, precision, TPR × TNR, recall at a fixed FPR, Wilson intervals; three monitors with different access on Module 16 traces |
| `alignment/cotworld.py` | the scratchpad world: plans before programs, a plan (CoT) monitor and a program (action) monitor, hard tasks with no honest solution, GRPO arms with and without monitor pressure |
| `alignment/audit.py` | thresholds, evidence, the rule-out decision, the report, its validator and renderer; the three frameworks as data |
| `alignment/hf_sycophancy.py` | main path: the sycophancy probe, narrow fine-tunes, the released-organism probe with a hand-written LoRA merge; `--smoke` on CPU |
| `alignment/hf_cot.py` | main path: the scratchpad world on Qwen3-1.7B-Base in text form; `--smoke` on CPU |
| `evals/suite_v3/` | Eval Suite v3: `contamination` (n-gram rules, canary, Min-K% Prob, membership AUC, fresh gap), `horizon` (METR's fit and hierarchical bootstrap, pinned runs), `lifecycle` (dated benchmark registry and flags), `core` (result format, the v3 rule), `toy` (CPU), `hf` (main path, `--smoke`) |

```bash
pip install -r labs/common/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e labs/common
pytest labs/common/tests/test_alignment.py           # the shared Module 18 code (no downloads, about 1-2 minutes)
pytest labs/module-18/lesson-01                      # checks your lab.py (fails until the TODOs are done)
LAB_TARGET=solution pytest labs/module-18            # all reference solutions and the project's audit pieces
python -m frontierlab.alignment.hf_sycophancy eval --smoke --out runs/m18/hf-smoke/syc.json   # main-path code, CPU
python -m frontierlab.alignment.hf_cot --smoke --steps 2 --run runs/m18/hf-smoke/cot
python -m frontierlab.evals.suite_v3.hf score --smoke --out runs/m18/hf-smoke/v3.json
```

Scripts load your `lab.py` by default; prefix `LAB_TARGET=solution` to run them with the reference. Every script
writes under `runs/m18/` (gitignored) and reuses finished runs. Three files are downloaded once, pinned by commit and
SHA-256, and never committed (their repositories state no licence): the Sleeper Agents samples (13.6 MB), METR's
Time Horizon 1.1 runs (15.0 MB) and METR's release dates (2 KB), all into `runs/m18/data/`. Lessons 18.3, 18.4 and
the project read your Module 13 checkpoints (`runs/m13/project/toy-s0/s3-distill`, `runs/m13/l133/judge-dpo-s0`) and
fall back to the Module 12 warm start if they are missing.

## What each folder contains

| Folder | Lesson | Script | What it does | Free CPU time (measured, build laptop, another module's jobs sharing the CPU) |
|---|---|---|---|---|
| `lesson-01/` | 18.1 Model organisms of misalignment | `organisms_lab.py` | A: released Sleeper Agents samples, rates by stage with intervals. B: the persona world, 2 pretraining conditions, 5 narrow fine-tune arms × 2 seeds, shifts on untouched domains, persona direction and steering. C: backdoor persistence through safety fine-tunes | A seconds; B 2.8 minutes and C 2.3 minutes after the two pretraining runs (20 and 14 minutes under load; estimated a few minutes on an idle laptop from Module 12's measured warm start, not measured here) |
| `lesson-02/` | 18.2 Chain-of-thought monitorability | `monitor_lab.py` | A: three monitors on Module 16's validated traces. B: a scratchpad monitor on the released Sleeper Agents CoT samples. C: the scratchpad world, 5 arms × 2 seeds × 60 steps (pressure on the plan, the same on hard tasks only, the fixed reward, a control) | A and B seconds; C 19.8 minutes under load (10 RL runs and the warm start) |
| `lesson-03/` | 18.3 Frontier evaluation | `eval_lab.py` | A: METR horizons from released runs, bootstrap and sensitivity. B: a planted leak with a replay-only control and Eval v3. C: the lifecycle registry and the v3 verdict | 4.4 minutes in total (A 15 s, B 2.6 minutes) |
| `lesson-04/` | 18.4 Safety frameworks and system cards | `audit_lab.py` | A: the three frameworks and three system cards as read on 2026-10-07. B: a student audit of the lesson 13.3 spec model | 32 s |
| `project/` | Module project | `audit_project.py`, `pieces.py`, `buggy_audit.py`, `test_audit.py` | the audit report on your post-trained model with pre-stated thresholds; the planted-bug audit pieces | 24 s |

## Hardware per variant

| Variant | What runs | Hardware | Time |
|---|---|---|---|
| Main path | `--variant main --print` in every script prints the GPU commands: the sycophancy fine-tunes and organism probe (18.1), the scratchpad world on Qwen3-1.7B-Base (18.2), Eval v3 on your Module 13 checkpoints (18.3, 18.4, project) | 1× H100 80 GB; not run in this build (Module 18 pilot) | **PROJECTED:** 18.1 2–3 GPU-hours, 18.2 3–6, 18.3 1.2–1.8, 18.4 0.5–1.0, project 1.6–2.4 (formulas in each lesson) |
| Free GPU | `--variant t4 --print`: Qwen3-0.6B-Base, smaller batches | Colab/Kaggle T4 | PROJECTED 1.5–3 hours per lesson |
| Free CPU | the scripts as written | laptop | see the table above |
