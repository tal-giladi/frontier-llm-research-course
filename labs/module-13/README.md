# Module 13 labs — Which post-training pipeline for which target?

Each lesson folder has `lab.py` (yours, with TODOs that raise `NotImplementedError("TODO n: ...")`),
`solution.py` (the reference), `test_lab.py` and the scripts the lesson runs. The shared code is
`labs/common/frontierlab/pipeline/`, with tests in `labs/common/tests/test_pipeline.py`; it builds on Module 12's
`frontierlab/posttrain/` (the toy world, the RL loop) and `frontierlab/evals/suite_v2/` (Eval Suite v2):

| Module | What it does |
|---|---|
| `pipeline/compute.py` | the FLOP ledger every arm reports: student sampling and training, reference, teacher sampling and scoring, judge labelling, training and scoring, router |
| `pipeline/seqs.py` | (prompt, response) examples, padding and response masks, sequence log-probabilities, the SFT loss and loop (from scratch) |
| `pipeline/dpo.py` | DPO from scratch: plain, length-normalised (Tülu 3), with an NLL term (Llama 3), soft labels; the cached reference pass |
| `pipeline/toy.py` | the toy world as pipeline data: two-aspect ratings, Tülu 3 binarisation, sampling with cost, Eval v2 |
| `pipeline/distill.py` | the teacher, exact and sampled reverse KL, on-policy distillation, SFT on teacher outputs, cost of a Module 12 RL run |
| `pipeline/judge.py` | Spec-T (the toy mini-spec), the simulated AI labeller, the judge model, judge reports, adherence |
| `pipeline/thinking.py` | the two-mode toy task, budget forcing, routing curves, random routing and the oracle |
| `pipeline/recipe_r.py` | the short pipeline on the Module 11 Recipe-R checkpoint (BPE tokenizer, validation-loss retention) |
| `pipeline/hf_eval.py` | Eval Suite v2 for Hugging Face checkpoints at pinned revisions, plain or chat format |
| `pipeline/hf_stages.py` | the main path on Qwen3-1.7B-Base: `sft`, `dpo`, `rlvr` (with a random-reward control), `distill`, `spec`, `spec-eval` |

```bash
pip install -r labs/common/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e labs/common
pytest labs/common/tests/test_pipeline.py               # the shared Module 13 code (no downloads, about a minute)
pytest labs/module-13/lesson-01                         # checks your lab.py (fails until the TODOs are done)
LAB_TARGET=solution pytest labs/module-13               # all reference solutions and the project's piece tests
```

Scripts load your `lab.py` by default; prefix `LAB_TARGET=solution` to run them with the reference. Every script
writes under `runs/m13/` (gitignored) and skips finished runs. The scripts need Module 12's SFT checkpoint
(`runs/m12/sft`, trained in one to three minutes if missing) and lesson 13.2's teacher (`runs/m13/teacher`, trained
once in 4–15 minutes if missing); the Recipe-R part of the project needs your Module 11 checkpoint.

## What each folder contains

| Folder | Lesson | Scripts | What they do |
|---|---|---|---|
| `lesson-01/` | 13.1 Open recipes | `recipe_lab.py` | four DPO arms (plain, length-normalised, + NLL, random-label control) on on-policy pairs, judged by Eval v2; the audit of published stage tables; main path: Eval v2 on the OLMo 2 1B stage checkpoints |
| `lesson-02/` | 13.2 Distillation | `distill_lab.py` | SFT on teacher outputs, on-policy distillation (sampled and exact), RL and a self-teacher control, all FLOPs counted; main path: Qwen3-8B into Qwen3-1.7B-Base |
| `lesson-03/` | 13.3 Specification-driven alignment | `spec_lab.py` | AI feedback, a judge and its error report; judge, judge+verifier, judge-SFT, oracle and random arms; adherence and Eval v2; main path: CoCoNot with a Qwen3-8B judge |
| `lesson-04/` | 13.4 Thinking modes (extension) | `think_lab.py`, `think_main.py` | a two-mode toy model with and without budget-aware data, budget forcing, a router against random and oracle routing; main path: Qwen3-1.7B on GSM8K |
| `project/` | Module project | `pipeline_lab.py`, `buggy_pipeline.py`, `pieces.py`, `test_pieces.py` | SFT → DPO → RLVR (+ random-reward control) → distillation with Eval v2 after each stage; the short pipeline on Recipe-R; the planted-bug task |

## Hardware and time per variant

Main-path commands were **not run in this build**; they are part of the Module 13 pilot, and every main-path figure
in the lessons is PROJECTED with its formula. Every main-path script has a `--smoke` mode that runs the same code on
tiny random models on a CPU. Free CPU times were measured on 2026-10-07 on a 16-thread Windows 11 laptop (torch
2.14.1+cpu, transformers 5.18.0, 4–8 threads per script) **with another module's jobs running**.

| Lab | Main path (1× H100, PROJECTED) | Free GPU (T4) | Free CPU (measured) |
|---|---|---|---|
| 13.1 recipes | 2–3.5 GPU-hours (7 Eval v2 runs on OLMo 2 1B) | same, fewer items | `recipe_lab.py` 3 min 35 s |
| 13.2 distillation | 5–7 GPU-hours | Qwen3-0.6B-Base student, Qwen3-1.7B teacher | `distill_lab.py` 511 s, plus the teacher once (909 s under load) |
| 13.3 spec | 2–3 GPU-hours | Qwen3-0.6B-Base policy, Qwen3-1.7B judge | `spec_lab.py` 452 s |
| 13.4 thinking | 1–1.5 GPU-hours | Qwen3-0.6B | `think_lab.py` 704 s |
| Project | see the project page | Qwen3-0.6B-Base | `pipeline_lab.py toy` 283 s for 2 seeds; `recipe-r` 463 s; `buggy_pipeline.py` 77 s, `--fixed` 57 s |

Main-path downloads (all pinned by revision in the scripts; data files checked by SHA-256): Qwen3-1.7B-Base
(3.4 GB), Qwen3-8B (16.4 GB, teacher and judge), Qwen3-4B (8 GB, evaluation judge), Qwen3-1.7B (4 GB, lesson 13.4),
the OLMo 2 1B stage checkpoints (about 6 GB each in float32), one shard each of
`allenai/tulu-3-sft-olmo-2-mixture-0225` (108 MB) and `allenai/olmo-2-0425-1b-preference-mix` (199 MB),
`allenai/coconot` (2.8 MB), GSM8K and IFEval (Module 12).
