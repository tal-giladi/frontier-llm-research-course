# Module 14 labs — Which RL objective, at what scale?

Each lesson folder has `lab.py` (yours, with TODOs that raise `NotImplementedError("TODO n: ...")`),
`solution.py` (the reference), `test_lab.py` and the script the lesson runs. The shared code is
`labs/common/frontierlab/rlscale/`, with tests in `labs/common/tests/test_rlscale.py`. It builds on Module 12's
loop and Eval Suite v2 (`frontierlab.posttrain`, `frontierlab.evals.suite_v2`), which it imports and never edits:

| Module | What it does |
|---|---|
| `rlscale/objectives.py` | GRPO, DAPO, Dr. GRPO, GSPO, GSPO-token, CISPO: one function each, one interface, the settings each paper pairs with it, `gradient_weights` (the per-token weight on grad log pi), `as_hook` for the Module 12 loop |
| `rlscale/runner.py` | runs the Module 12 loop with an objective, a random or format control reward (held-out evaluation stays strict), or a bounded-staleness sampler; objective arms over seeds |
| `rlscale/stability.py` | the pre-stated stability metrics from a run's `metrics.jsonl` |
| `rlscale/asyncsim.py`, `mismatch.py` | the synchronous / bounded-staleness schedule model; trainer/sampler mismatch statistics and a batch-invariant matmul |
| `rlscale/curves.py`, `passk.py` | ScaleRL's sigmoid, its fit and the profile interval of the asymptote; pass@k curves, paired differences, crossover, solved-ever sets |
| `rlscale/hf_rl.py`, `vllm_rollout.py` | the main path: objective-switchable RL on Qwen3-1.7B-Base and GSM8K, control arms, pass@k evaluation, vLLM 0.30.0 rollouts (colocated, CUDA-IPC weight sync, optional batch-invariant mode, one-step off-policy) |

```bash
pip install -r labs/common/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e labs/common
pytest labs/common/tests/test_rlscale.py              # the shared Module 14 code (no downloads, about 1 minute)
pytest labs/module-14/lesson-01                       # checks your lab.py (fails until the TODOs are done)
LAB_TARGET=solution pytest labs/module-14             # all reference solutions and the project's objective tests
python -m frontierlab.rlscale.hf_rl --smoke --run runs/m14/hf-smoke --steps 2    # main-path code on a tiny random Qwen3, CPU
```

Scripts load your `lab.py` by default; prefix `LAB_TARGET=solution` to run them with the reference. Every
script writes under `runs/m14/` (gitignored), skips finished runs and resumes interrupted ones exactly
(the Module 12 loop's checkpoints). Every CPU run starts from the Module 12 SFT checkpoint (`runs/m12/sft`),
which the first script trains if it is missing (one to three minutes).

## What each folder contains

| Folder | Lesson | Script | What it does |
|---|---|---|---|
| `lesson-01/` | 14.1 Objectives derived, not switched | `objectives_lab.py` | your five losses on the worked example next to the hand values; each in the course loop for 80 steps with 2 epochs × 4 minibatches (a smoke test with diagnostics, not a comparison) |
| `lesson-02/` | 14.2 A controlled objective comparison | `compare_lab.py` | equal tuning budget (3 lr × 60 steps per objective), then 5 objectives + a random-reward control × 3 seeds × 120 steps; decision table, stability flags, Eval v2 guards |
| `lesson-03/` | 14.3 Rollout systems and staleness | `async_lab.py` | the schedule model on three generation/update balances; CPU batch invariance and the bf16-sampler mismatch; GRPO/CISPO/GSPO at staleness bound 8 vs on-policy vs uncorrected |
| `lesson-04/` | 14.4 RL scaling and the capability debate | `scaling_lab.py` | fit windows and profile intervals on a ScaleRL-shaped curve; your own 200-step run's asymptote; pass@k to k = 256 for the start, GRPO and a random-reward control |
| `project/` | Module project | `run_project.py`, `test_objectives.py`, `buggy_objectives.py`, `buggy_run.py` | the reasoning-RL run with stability, Eval v2 and pass@k; the objective derivation tests; the planted-bug debugging task |

## Hardware and time per variant

Main-path commands were **not run in this build**; they are part of the Module 14 pilot, and every main-path
figure in the lessons is PROJECTED with its formula. Free CPU times were measured on 2026-10-07 on a 16-thread
Windows 11 laptop (torch 2.14.1+cpu, transformers 5.18.0, 8 threads per script) **with another module's jobs
and a second Module 14 script running at the same time**, so expect them to be roughly half on an idle machine.

| Lab | Main path (1× H100, PROJECTED) | Free GPU (T4) | Free CPU (measured) |
|---|---|---|---|
| 14.1 objectives | 2.1–3.5 GPU-hours (5 × 50 steps) | Qwen3-0.6B-Base, `--variant t4 --print` | `objectives_lab.py` 6.9 min |
| 14.2 comparison | 31–52 GPU-hours (15 tuning + 12 comparison runs + Eval v2) | GRPO and CISPO only | `compare_lab.py` 35 min (33 runs + 18 Eval v2 scorings) |
| 14.3 staleness | 2.8–7 GPU-hours (5 × 100 steps, vLLM 0.30.0) | HF rollouts only | `async_lab.py` 19 min (staleness 18 min; schedule and mismatch under 1 min) |
| 14.4 scaling | 12–21 GPU-hours (2 × 300 steps + pass@k sampling) | Qwen3-0.6B-Base | `scaling_lab.py` about 7 min (two 200-step runs, fits, 153,600 samples) |
| Project | 25–47 GPU-hours | Qwen3-0.6B-Base | `run_project.py` 18.7 min; `buggy_run.py` 4.5 min |

Notes:

- Main path downloads: Qwen3-1.7B-Base (3.4 GB), Qwen3-0.6B-Base (1.2 GB), GSM8K (2.7 MB, SHA-256 checked). The
  vLLM path needs `pip install vllm==0.30.0 requests` on the GPU machine. It starts `vllm serve` itself, with
  `--weight-transfer-config '{"backend": "ipc"}'` and `VLLM_SERVER_DEV_MODE=1`, and sets
  `VLLM_ALLOW_INSECURE_SERIALIZATION=1` (needed by CUDA IPC handles over HTTP; use it only on a machine you control).
- On a T4 use fp32 and the 0.6B model; vLLM next to training does not fit at useful sizes on 16 GB.
- Run `python -m frontierlab.rlscale.hf_rl --smoke ...` (CPU, seconds) before any GPU session to check the install.
