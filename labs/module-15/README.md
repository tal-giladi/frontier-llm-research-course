# Module 15 labs — How should compute be spent at inference?

Each lesson folder has `lab.py` (yours, with TODOs that raise `NotImplementedError("TODO n: ...")`),
`solution.py` (the reference), `test_lab.py` and the script the lesson runs. The shared code is
`labs/common/frontierlab/ttc/`, with tests in `labs/common/tests/test_ttc.py`. It imports Modules 2–14 code
(`frontierlab.perf`, `attention.accounting`, `blocks.mtp`, `blocks.train`, `posttrain`, `pipeline`,
`evals.suite_v2`, `precision`, `calc`) and edits none of it:

| Module | What it does |
|---|---|
| `ttc/select.py` | majority vote, weighted vote, best-of-N (ties never broken by correctness), success of a procedure on random subsets of a pool, oracle pass@k, the selection gap |
| `ttc/budget.py` | the spend of a strategy (prefill, decode and verifier tokens, FLOPs, policy-token equivalents) and a latency model built from measured step times |
| `ttc/world.py` | the free-CPU world: four-digit addition with three reasoning efforts, budget forcing, an ORM and a PRM trained on the policy's samples, a programmatic checker, PRM-guided beam search, step-time measurement |
| `ttc/report.py` | budget-matched rows, best per family, the pre-stated decision rule |
| `ttc/speculative.py`, `ttc/eagle.py`, `ttc/spec_models.py` | speculative sampling (batched accept/reject, residual, cache rollback), independent-LM and hidden-state drafts, the speed-up formulas; an EAGLE-style head trained on a frozen target; the 15.2/15.4 toy models |
| `ttc/serving.py` | roofline prefill and decode, KV bytes of the Module 3–5 designs and released models, capacity, an iteration-level serving simulator (prefill-first, chunked, disaggregated), RL rollout time |
| `ttc/kvquant.py` | asymmetric group quantisation, KIVI's cache layout as the attention kind `"gqa-kvq"`, decoding NLL/KL with a quantised cache, weight RTN |
| `ttc/hf_ttc.py`, `ttc/hf_spec.py` | the main path: test-time compute on GSM8K with Qwen3-1.7B, vLLM 0.30.0 or Transformers, ORM and PRM verifiers; speculative decoding on Qwen3-1.7B-Base with a Qwen3-0.6B-Base draft and through vLLM's `speculative_config` |

```bash
pip install -r labs/common/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e labs/common
pytest labs/common/tests/test_ttc.py                 # the shared Module 15 code (no downloads, about 1 minute)
pytest labs/module-15/lesson-01                      # checks your lab.py (fails until the TODOs are done)
LAB_TARGET=solution pytest labs/module-15            # all reference solutions and the project's harness tests
python -m frontierlab.ttc.hf_ttc smoke --out runs/m15/hf-smoke                     # main-path code on tiny random models, CPU
python -m frontierlab.ttc.hf_spec own --smoke --out runs/m15/l152-smoke
```

Scripts load your `lab.py` by default; prefix `LAB_TARGET=solution` to run them with the reference. Every script
writes under `runs/m15/` (gitignored) and reuses what it has already trained or sampled. Lessons 15.2 and 15.4 need
Data-v0 (`python -m frontierlab.data.prepare --docs 20000 --vocab 8192`, Module 1). Lesson 15.4 reuses lesson
15.2's target model and trains it if it is missing.

## What each folder contains

| Folder | Lesson | Script | What it does |
|---|---|---|---|
| `lesson-01/` | 15.1 A test-time compute experiment | `ttc_lab.py` | trains the policy and two verifiers, samples pools, runs PRM beam search, measures step times; the selection-gap table (your functions), every strategy at matched spend with latency, the recommendation at five budgets |
| `lesson-02/` | 15.2 Speculative decoding | `spec_lab.py` | trains the target (with MTP), a draft LM and an EAGLE-style head; greedy equivalence check with your `accept_reject`; acceptance by draft, gamma, temperature and position; measured c, predicted and measured speed-ups |
| `lesson-03/` | 15.3 Serving cost of architecture choices | `serve_lab.py` | KV at 128K for the designs and released models; roofline tables; CPU measurements and cache checks; the serving simulator on 4 GPUs; RL rollout time per design |
| `lesson-04/` | 15.4 KV and weight quantisation (extension) | `quant_lab.py` | outlier statistics of the cache; KV formats decoded with the cache (NLL change with paired intervals, KL); weight RTN formats; capacity per format at 128K |
| `project/` | Module project | `run_project.py`, `harness.py`, `buggy_harness.py`, `test_harness.py`, `buggy_run.py` | the 15.1 experiment for two seeds and three latency targets; the planted-bug debugging task |

## Hardware and time per variant

Main-path commands were **not run in this build**; they are part of the Module 15 pilot, and every main-path
figure in the lessons is PROJECTED with its formula. Free CPU times were measured on 2026-10-07 on a 16-thread
Windows 11 laptop (torch 2.14.1+cpu, transformers 5.18.0, 8 threads per script) **with another module's jobs
running at the same time**, so expect them to be shorter on an idle machine.

| Lab | Main path (1× H100, PROJECTED) | Free GPU (T4) | Free CPU (measured) |
|---|---|---|---|
| 15.1 test-time compute | 2–4 GPU-hours (sampling, single chains, ORM scoring, PRM search, latency) | Qwen3-0.6B, no PRM arm | `ttc_lab.py` about 10 min the first time (policy 1.2 min, ORM 1.5 min, PRM 2.3 min), 20–30 s from the cache |
| 15.2 speculative decoding | under 1 GPU-hour | fp16, 16 prompts | `spec_lab.py` 17 min the first time (target 11 min, draft 2 min, head 2 min), 2 min after |
| 15.3 serving cost | under 0.5 GPU-hours (two `vllm bench throughput` runs) | `--input-len 2048` | `serve_lab.py` about 2–3 min |
| 15.4 quantisation | under 1 GPU-hour | emulation on the GPU | `quant_lab.py` 103 s after lab 15.2 |
| Project | 4–7 GPU-hours | Qwen3-0.6B | `run_project.py` 77 s after lab 15.1; `buggy_run.py` seconds |

Notes:

- Main path downloads: Qwen3-1.7B (4.1 GB), Qwen3-1.7B-Base (3.4 GB), Qwen3-0.6B-Base (1.2 GB),
  Skywork-Reward-V2-Qwen3-1.7B (3.4 GB), Qwen2.5-Math-PRM-7B (15 GB), AngelSlim/Qwen3-1.7B_eagle3 (small; custom
  licence), GSM8K (2.7 MB, SHA-256 checked). Install `pip install vllm==0.30.0` on the GPU machine.
- Qwen2.5-Math-PRM-7B loads with `trust_remote_code=True` at a pinned revision: read the remote code at that
  revision before running it, and run it only on a machine you control. Its licence is the Qwen licence.
- The PRM (15 GB in BF16) and vLLM with the policy fit together on one 80 GB GPU with
  `gpu_memory_utilization=0.45`; on a T4 they do not, so the T4 variant has no search arm.
- Run the `--smoke` commands (CPU, seconds to minutes) before any GPU session to check the install.
