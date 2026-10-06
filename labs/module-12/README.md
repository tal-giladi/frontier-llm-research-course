# Module 12 labs — Post-training foundations

Each lesson folder has `lab.py` (yours, with TODOs that raise `NotImplementedError("TODO n: ...")`),
`solution.py` (the reference), `test_lab.py` and the scripts the lesson runs. The shared code is
`labs/common/frontierlab/posttrain/` and `labs/common/frontierlab/evals/suite_v2/`, with tests in
`labs/common/tests/test_posttrain.py`:

| Module | What it does |
|---|---|
| `posttrain/tokenizer.py`, `tasks.py` | the toy verifiable world: a 35-token character tokenizer; 2-digit arithmetic with instruction tags; strict, last-number and lenient verifiers; the letter world with a hidden gold reward (12.1) |
| `posttrain/policy.py` | sampling with the KV cache, teacher-forced log-probabilities, entropy, the response mask, exact token KL |
| `posttrain/sft.py` | the supervised warm start (loss on response tokens only; stops at about 30% plain-addition accuracy) |
| `posttrain/reward.py` | Bradley-Terry reward models, calibration (ECE), best-of-n KL and the unbiased BoN estimator, Gao et al. fits |
| `posttrain/advantages.py`, `kl.py` | baselines, group and batch normalisation, zero-variance groups, GAE; k1/k2/k3 and their exact expected gradients |
| `posttrain/losses.py` | token and sequence ratios, clipping, four aggregation modes, DAPO's overlong penalty and filter, truncated IS |
| `posttrain/rl.py` | the RL loop (`python -m frontierlab.posttrain.rl`): every detail a flag, every step logged, exact resume, hooks for lab code |
| `posttrain/arms.py` | runs arms over seeds and compares them paired by seed |
| `posttrain/gsm8k.py`, `hf.py` | the main path: GSM8K at a pinned revision, and the same loop on a Hugging Face model (Qwen3-1.7B-Base) |
| `evals/suite_v2/` | Eval Suite v2: `core.py` (pass@k, paired comparison, retention guard), `toy.py` (the CPU suite), `ifeval.py` (12 of IFEval's 25 instruction types, pinned data) |

```bash
pip install -r labs/common/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e labs/common
pytest labs/common/tests/test_posttrain.py            # the shared Module 12 code (no downloads, about 30 s)
pytest labs/module-12/lesson-01                       # checks your lab.py (fails until the TODOs are done)
LAB_TARGET=solution pytest labs/module-12             # all reference solutions and the project's detail tests
```

Scripts load your `lab.py` by default; prefix `LAB_TARGET=solution` to run them with the reference. Every
script writes under `runs/m12/` (gitignored) and skips finished runs; an interrupted RL run resumes exactly
when you rerun the same command. The first script you run trains the SFT warm start (`runs/m12/sft`, one to
three minutes); every later lab starts from it.

## What each folder contains

| Folder | Lesson | Scripts | What they do |
|---|---|---|---|
| `lesson-01/` | 12.1 Rewards | `reward_lab.py`, `rm_main.py` | verifier audit (probe set, policy samples, optional RL against a lenient verifier); reward models on 250/1,000/4,000 pairs and best-of-n against a hidden gold reward; main path: Gao et al.'s synthetic-gold setup with Qwen3 models |
| `lesson-02/` | 12.2 Policy-gradient estimators | `estimator_lab.py` | REINFORCE vs GRPO vs RLOO vs Dr. GRPO (3 seeds); KL placements and estimators against the exact KL (2 seeds); an entropy bonus |
| `lesson-03/` | 12.3 Details that change results | `details_lab.py` | aggregation modes; truncation handling under a tight budget; three loss masks; staleness and a bf16 sampler with and without truncated IS |
| `lesson-04/` | 12.4 Eval Suite v2 | `eval_lab.py`, `eval_main.py` | three RL arms scored by Eval v2 against the SFT start, pass@k curves; main path: GSM8K, LAMBADA and the IFEval subset for a Hugging Face checkpoint |
| `project/` | Module project | `test_loop.py`, `buggy_loop.py` | one test per 12.2–12.3 detail; the planted-bug debugging task |

## Hardware and time per variant

Main-path commands were **not run in this build**; they are part of the Module 12 pilot, and every main-path
figure in the lessons is PROJECTED with its formula. Free CPU times were measured on 2026-10-07 on a 16-thread
Windows 11 laptop (torch 2.14.1+cpu, transformers 5.18.0, 8 threads per script) **with another module's jobs
running**, so expect them to be shorter on an idle machine.

| Lab | Main path (1× H100, PROJECTED) | Free GPU (T4) | Free CPU (measured) |
|---|---|---|---|
| 12.1 rewards | 2–3 GPU-hours (`rm_main.py`) | `rm_main.py --gold 1.7b`, smaller | `reward_lab.py --rl-lenient` 9 min |
| 12.2 estimators | 8–12 GPU-hours (8 runs of 100 steps) | Qwen3-0.6B-Base | `estimator_lab.py` 20 min (25 runs) |
| 12.3 details | 12–18 GPU-hours (12 runs) | Qwen3-0.6B-Base | `details_lab.py` 21 min (30 runs) |
| 12.4 Eval v2 | 5–7 GPU-hours | `eval_main.py` on 0.6B | `eval_lab.py` 7.5 min |
| Project | 9–14 GPU-hours | Qwen3-0.6B-Base | `test_loop.py` about 10 s; `buggy_loop.py` 95 s, `--fixed` 88 s |

Notes:

- Main path downloads: Qwen3-1.7B-Base (3.4 GB in BF16), Qwen3-0.6B-Base (1.2 GB), Skywork-Reward-V2-Qwen3-8B
  (16 GB; lesson 12.1 only), GSM8K (2.7 MB), IFEval (207 KB), UltraFeedback prompts (lesson 12.1). All pinned
  by revision in the scripts; GSM8K and IFEval files are checked by SHA-256.
- On a T4 use fp32 and the 0.6B base model; the 1.7B model trains in fp32 with AdamW only on a large GPU.
- `python -m frontierlab.posttrain.hf --smoke` and `eval_main.py --smoke` / `rm_main.py --smoke` run the
  main-path code on tiny random models on a CPU in under a minute: use them to check an install.
