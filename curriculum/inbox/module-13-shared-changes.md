# Module 13 — proposed changes to shared files (for the main session)

Module 13 does not edit any existing `frontierlab` file, `_sidebar.md`, `glossary.md`, `references/` or `templates/`.
New code lives in `labs/common/frontierlab/pipeline/` (a new subpackage; nothing needs to import it), tests in
`labs/common/tests/test_pipeline.py`. It imports `frontierlab.posttrain` and `frontierlab.evals.suite_v2` read-only.

## 1. `_sidebar.md`: Module 13 lines

Running numbers 55–58 (after Module 12's 51–54). Lesson 13.4 is an extension.

```markdown
- **Module 13 — Which post-training pipeline for which target?**
  - [55 · Open recipes as case studies](lessons/module-13/lesson-01.md)
  - [56 · Distillation](lessons/module-13/lesson-02.md)
  - [57 · Specification-driven alignment](lessons/module-13/lesson-03.md)
  - [58 · Thinking modes and budgets](lessons/module-13/lesson-04.md)
  - [Module 13 quiz](assessments/module-13-quiz.md)
```

The project page `projects/module-13-pipeline.md` is picked up from `projects/` automatically.

## 2. `glossary.md`

Merge `curriculum/glossary-inbox/module-13.md` (31 terms).

## 3. `references/versions.md`: Module 13 models, data and framework notes (checked 2026-10-07)

Add a section "Module 13 models and data (checked 2026-10-07)":

| Item | Revision | Licence | Used in |
|---|---|---|---|
| Qwen/Qwen3-8B (distillation teacher, training judge; 8,190,735,360 parameters, BF16) | `b968826d9c46dd6066d109eabc6255188de91218` | Apache-2.0 | 13.2, 13.3, project main path |
| Qwen/Qwen3-4B (evaluation judge) | `1cfa9a7208912126459214e8b04321603b3df60c` | Apache-2.0 | 13.3 main path |
| Qwen/Qwen3-1.7B (hybrid thinking; 2,031,739,904 parameters in the safetensors, output matrix stored separately) | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Apache-2.0 | 13.4 main path; 13.2 T4 teacher |
| allenai/OLMo-2-0425-1B-SFT | `0d85a3d037876ce6ac7d4311d994400fc66ac27f` | Apache-2.0 | 13.1 main path |
| allenai/OLMo-2-0425-1B-DPO | `c4b0485961ab24c2433b090f3b922f0913a9290f` | Apache-2.0 | 13.1 main path |
| allenai/OLMo-2-0425-1B-Instruct (RLVR stage) | `48d788eca847d4d7548f375ad03d3c9312f6139e` | Apache-2.0 | 13.1 main path |
| allenai/tulu-3-sft-olmo-2-mixture-0225, `data/train-00004-of-00006.parquet` (108,173,855 B) | `d91a0785ade02942520280fb484866fce41e448f` | ODC-BY (subsets vary) | project main path (SFT) |
| allenai/olmo-2-0425-1b-preference-mix, `data/train-00004-of-00005.parquet` (199,266,481 B) | `c6454f3d364622718a8ecd3d307e524febbba569` | ODC-BY 1.0, some subsets non-commercial | project main path (DPO) |
| allenai/coconot `original/train` (11,477), `original/test` (1,001), `contrast/test` (379) | `2cbe16aabf9069f17e48c8daad8aeabc29469eb7` | ODC-By | 13.3 main path |

(The full SHA-256 values are in `labs/common/frontierlab/pipeline/hf_stages.py`, `DATA`.)

Add to "Notes on reference implementations":
"Module 13: TRL v1.14.1 `DPOConfig`: `loss_type` is a list, default `["sigmoid"]`; `"sigmoid_norm"` divides each
log-ratio by its response length (Tülu 3's length-normalised DPO); an NLL term on the chosen responses is
`loss_type=["sigmoid", "sft"]` with `loss_weights` (TRL's `sft` term is a token mean over the batch's chosen tokens);
`precompute_ref_log_probs=True` caches the reference (`trl/trainer/dpo_config.py`, `dpo_trainer.py` around lines
1570–1590). GKD is `trl.experimental.gkd` (`GKDConfig(lmbda=0.5, beta=0.5)`; per its docstring `beta=0.0` = KL,
`beta=1.0` = inverse KL). No `rpo_alpha` field exists at this tag."

## 4. Changes wanted in `frontierlab.posttrain` (owned by Module 12; not made)

1. `posttrain/hf.py`: a `reward_fn` hook (or `HFConfig.reward = "strict" | "random"`) so a random-reward control
   arm does not need the monkeypatch in `pipeline/hf_stages.py cmd_rlvr` (which replaces `gsm8k.strict_reward`
   for the whole run, so the control's in-run `pass` and eval are the random reward; it must be judged by Eval v2
   afterwards). Proposed: `def train(cfg, reward_fn=None)` with `reward_fn(decoded, answer) -> float`, defaulting
   to `gsm8k.strict_reward`, used only for the training reward, not for `lenient_pass` or the eval.
2. `posttrain/policy.py`: an `eos_id` argument to `sample` (default `EOS`). With another tokenizer (Recipe-R's BPE,
   where id 2 is an ordinary token) the sampler never stops; `pipeline/recipe_r.py` carries its own copy for that
   reason.
3. `posttrain/rl.py`: `RLConfig.init` accepts only the toy `save_policy` format; fine for Module 13.

## 5. requirements

No new packages. Module 13 uses `pyarrow` (parquet, already present), `tokenizers` (Recipe-R) and `transformers`
(smoke tests of the main path).

## 6. Plan / claim checks to record (plan section 14.1)

- V (row 28, refined): Thinking Machines on-policy distillation: student Qwen3-8B-Base, teacher Qwen3-32B, from a
  400k-example SFT checkpoint at 60% AIME'24; about 150 steps (77K prompts × 4 samples) to 70%, against about 2M
  examples extrapolated for SFT; 9× "when the SFT dataset is given", about 18× in GPU-hours counting teacher
  log-prob FLOPs for on-policy only, about 30× including teacher sampling for off-policy. The "7–10× faster than
  RL" figure is for their LoRA rank-128 setting. Advantage = −reverse KL, discount factor zero.
- V: Qwen3 Table 21 (8B): off-policy distilled 55.0 AIME'24 (pass@64 90.0); + RL 67.6 (90.0) in 17,920 GPU-hours;
  + on-policy distillation 74.4 (93.3) in 1,800 GPU-hours. Section 4.3 stop-thinking text and "not explicitly
  trained but emerges naturally". Table 22 (32B): ThinkFollow 88.7 → 98.9; AIME'24 thinking 83.8 → 81.9 → 81.4.
- V: Tülu 3: 939,344 SFT prompts (Table 7); 354,192 preference instances (8B); GPT-4o ratings 1–5 on four aspects,
  binarisation rule (5.2.1); length-normalised DPO Eq. 6 (5.1.2), β = 5, lr 5e-7 for 8B (Table 20); only
  length-normalised DPO beat the base checkpoint in Table 18; RLVR α = 10, β = 0.05 (Table 21); 8B average SFT 60.6,
  DPO 64.7, final 65.1 (Table 6); SFT seeds 59.8–60.1 (Table 14); final checkpoint chosen on MATH and IFEval (6.4).
- V: Olmo 3 (2512.13961): delta learning pairs (chosen Qwen3 32B thinking, rejected Qwen3 0.6B thinking, 4.3.1);
  further SFT on Qwen3 32B traces "outright hurts"; OlmoRL objective Eq. 1–2 (token-level loss, TIS, clip-higher,
  no std); Table 22 one run: SFT 70.1, +DPO 72.7, +RLVR 74.1, SFT+RLVR 71.9.
- V: Llama 3 4.1.4: DPO β 0.1, lr 1e-5, formatting tokens masked, NLL 0.2; six rounds (4.1.6); rejection sampling
  K = 10–30 (4.2.2).
- V: Constitutional AI: 16 principles, 4 revisions per red-team prompt (3); multiple-choice soft labels (4.1);
  CoT labels clamped 40–60% (4.3); "harmless but non-evasive"; Goodharting when over-trained.
- V: Deliberative alignment: spec-aware judge, minimum of k scores (2.3.2), spec removed from the prompt (2.3.3), CoT
  hidden from the judge in RL (2.4), SFT-only and RL-only ablations intermediate (4.1).
- V (company documents): Claude's constitution 2026-01-22, priority order and "generally prioritize"; Model Spec
  2026-08-18 levels Root/System/Developer/User/Guideline; GPT-5 system card section 1 router sentence.
- S: GPT-5.1 addendum (openai.com returned 403; quoted via simonwillison.net); the Qwen team's July 2025 statement on
  dropping hybrid thinking (via The Register; the X post itself not fetched).

## 7. Colab / H100 pilot notebook: Module 13 commands (not run in this build)

From the repo root with `LAB_TARGET=solution`. First run the CPU smoke tests of the same code paths:
`python -m frontierlab.pipeline.hf_stages {sft,dpo,distill --mode onpolicy,spec,spec-eval,rlvr --control random} --smoke --model x --student x --teacher y --out runs/m13/hf-smoke/<name>`,
`python -m frontierlab.pipeline.hf_eval score --smoke --out runs/m13/hf-eval-smoke/a.json`,
`python labs/module-13/lesson-04/think_main.py --smoke --out runs/m13/l134-smoke`.
Record for each: wall time, GPU type, peak memory, generated tokens per second, and the printed tables.

| Lesson | Command | PROJECTED cost (H100) | Pilot question |
|---|---|---|---|
| 13.1 | `python labs/module-13/lesson-01/recipe_lab.py --variant main --print` (7 `hf_eval score` runs + 3 compares on OLMo 2 1B base/SFT/DPO/Instruct) | 2–3.5 GPU-h (0.3–0.5 per checkpoint) | which released stage gains exceed the paired item interval; does the chat format change the stage ranking? |
| 13.2 | `python labs/module-13/lesson-02/distill_lab.py --variant main --print` (offline and on-policy distillation Qwen3-8B → Qwen3-1.7B-Base, RL, self-teacher control, Eval v2 each) | 5–7 GPU-h | does on-policy distillation beat SFT on teacher outputs at 300-token responses? measured teacher-scoring vs student-sampling time |
| 13.3 | `python labs/module-13/lesson-03/spec_lab.py --variant main --print` (`hf_stages spec` with a Qwen3-8B judge, `spec-eval` with Qwen3-4B, Eval v2) | 2–3 GPU-h | judge agreement with the 50-response human audit; over-refusal on CoCoNot contrast before and after |
| 13.4 | `python labs/module-13/lesson-04/think_main.py --model Qwen/Qwen3-1.7B --n 500 --budgets 0,256,512,1024,2048,none --out runs/m13/l134-main`, then `--route cascade` | 1–1.5 GPU-h | is accuracy monotone in the budget for a 1.7B hybrid model? cascade cost at matched accuracy |
| project | `python labs/module-13/project/pipeline_lab.py main --print` (SFT 1,000 steps, DPO 300, RLVR 200 + random control, on-policy distillation 100, Eval v2 after each) | 14–22 GPU-h | per-stage Eval v2 deltas on the Stage D base; does the random-reward control move GSM8K on Qwen3-1.7B-Base (Shao et al.'s Qwen effect)? |

Scaled pilot per plan section 12.1 (row "12–14 RL loop"): run the project main path on Qwen3-0.6B-Base first
(`--model Qwen/Qwen3-0.6B-Base` for `sft`; `--max-new 256` for RLVR and distillation), 1 seed, and project the 1.7B
cost as (1.72/0.60) × measured training time + measured generation time × (1.7B decode time / 0.6B decode time),
labelled PROJECTED.

## 8. BUILD_PROGRESS / TODO_FOR_TAL

- Free-CPU timings in Module 13 were measured with Module 14's jobs sharing the CPU.
- The Recipe-R part of the project uses the Module 11 CPU target checkpoint `runs/m11/project/cpu/target-m11-r5-x10`
  (learners use their own).
