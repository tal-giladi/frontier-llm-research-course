# Module 12 — proposed changes to shared files (for the main session)

Module 12 does not edit any existing `frontierlab` file, `_sidebar.md`, `glossary.md`, `references/` or
`templates/`. New code lives in `labs/common/frontierlab/posttrain/` and `labs/common/frontierlab/evals/suite_v2/`
(a new subpackage of `evals`; `frontierlab/evals/__init__.py` is unchanged and does not need to import it).
Tests: `labs/common/tests/test_posttrain.py`. The RL loop is its own module (`python -m frontierlab.posttrain.rl`),
not an extension of `frontierlab.train.loop` (RL steps are not LM steps), so no loop change is needed.

## 1. `_sidebar.md`: Module 12 lines

Running numbers continue after Module 11; the brief reserves 51–54 (fix them if Module 11 ends elsewhere).
A stage line for Stage D goes before the module line.

```markdown
- Stage D — Post-training questions
- **Module 12 — Post-training foundations**
  - [51 · Rewards](lessons/module-12/lesson-01.md)
  - [52 · Policy-gradient estimators for LLMs](lessons/module-12/lesson-02.md)
  - [53 · Details that change results](lessons/module-12/lesson-03.md)
  - [54 · Eval Suite v2](lessons/module-12/lesson-04.md)
  - [Module 12 quiz](assessments/module-12-quiz.md)
```

The project page `projects/module-12-rl-loop.md` is picked up from `projects/` automatically.

## 2. `glossary.md`

Merge `curriculum/glossary-inbox/module-12.md` (34 terms).

## 3. `references/versions.md`: the Stage D base model and Module 12 data and models

Add a section "Stage D base model and Module 12 data and models (checked 2026-10-06)":

| Item | Revision | Licence | Used in |
|---|---|---|---|
| **Qwen/Qwen3-1.7B-Base (Stage D base model)** | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` | Apache-2.0 | Modules 12–16 main path (1,720,574,976 parameters, BF16) |
| Qwen/Qwen3-0.6B-Base (T4 variant; 12.1 proxy reward models) | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | Apache-2.0 | 12.1–12.4 |
| Skywork/Skywork-Reward-V2-Qwen3-8B (12.1 gold reward) | `6f19fdefb933293d4898bdb59a96f7223d998659` | Apache-2.0 | 12.1 main path |
| Skywork/Skywork-Reward-V2-Qwen3-1.7B (12.1 gold reward, T4) | `e51ea3e08fb81326c3b812a7ff0cb9cee83e59cc` | Apache-2.0 | 12.1 T4 |
| openai/gsm8k `main` (train 2,306,545 B, test 419,088 B; SHA-256 = LFS oids in `posttrain/gsm8k.py`) | `740312add88f781978c0658806c59bc2815b9866` | MIT | 12.2–12.4 main path |
| google/IFEval `ifeval_input_data.jsonl` (207,111 B, 541 prompts, SHA-256 `6a85310c…88f2`) | `966cd89545d6b6acfd7638bc708b98261ca58e84` | Apache-2.0 | 12.4 (Eval v2) |
| HuggingFaceH4/ultrafeedback_binarized (prompts only) | `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85` | MIT | 12.1 main path |

Proposed decision note for `BUILD_PROGRESS.md` / plan section 15: "Stage D base model: Qwen3-1.7B-Base
(`ea980cb`, Apache-2.0), dense Qwen3 layout identical to Baseline-0's; documented alternative OLMo-2-0425-1B
(`a1847dff35000b4271fa70afc5db10fd29fedbdf`, Apache-2.0, open data) for contamination-sensitive studies.
Considered and rejected: Qwen3.5-2B-Base (hybrid linear attention + vision encoder). Because of the
spurious-reward results on Qwen2.5-Math (Shao et al. 2506.10947), Stage D RL results include a random- or
format-reward control arm." Tal may want to confirm this choice.

Framework names in lesson 12.2's mapping table were read from source at TRL v1.14.1 and verl v0.9.1 (the
pinned versions) on 2026-10-07. Worth adding to the "Notes on reference implementations" section:
"Module 12: TRL v1.14.1 `GRPOConfig`: `loss_type` default `"dapo"`, `scale_rewards` `group|batch|none`,
`beta` default 0.0 (k3 per token, times the ratio when `use_bias_correction_kl=True`),
`vllm_importance_sampling_correction=True` with `vllm_importance_sampling_clip_max=3.0`. verl v0.9.1:
`kl_loss_type` `kl|abs|mse|low_var_kl|full` (+ suffix = k2 gradient), `loss_agg_mode`
`token-mean|token-sum|seq-mean-token-sum|seq-mean-token-mean|seq-mean-token-sum-norm`,
`algorithm.rollout_correction.rollout_is` / `rollout_is_threshold` (2.0)."

## 4. `requirements-cpu.txt`

No new packages. Module 12 uses `scipy` (already installed; t-intervals, `gammaln`), `pyarrow` (with
`datasets`, for the GSM8K parquet) and `transformers` (Qwen3 classes for the main-path smoke tests).

## 5. Plan / claim checks to record (plan section 14.1)

- C: DAPO overlong shaping: Eq. 13's `L_max` is the generation budget, 20,480 = 16,384 ("expected maximum
  length") + 4,096 (cache) (section 4.1). This closes the open gap on row 12.
- V: Gao et al. KL penalty "does not affect the KL_RL-gold reward frontier" (section 3.6).
- V: Tang & Munos (2506.09477, section 3): k1-as-loss zero-mean, k2-as-loss reverse-KL gradient,
  k3-as-loss forward-KL gradient; also checked exactly by enumeration in `test_posttrain.py`.
- V: Shao et al. spurious rewards: Qwen2.5-Math-7B MATH-500 +21.4 random, +24.1 incorrect, +29.1 ground
  truth (section 2.2); no gains on Llama 3 / OLMo 2 (section 3).
- V: Tülu 3 RLVR α = 10 (Eq. 8), β = 0.05 for the final 8B (Table 21).
- P: GSPO equation numbering (definition of s_i and the objective) differs between HTML renderings; the
  lessons cite section 4.1 rather than an equation number.
- P: ScaleRL's "prompt average (as in DAPO)" wording; the course implements prompt average as "each prompt
  equal, tokens equal within a prompt" and labels it as its reading.
- Not found: an arXiv id for Yao et al.'s TIS work (NeurIPS 2025 version "On the Rollout-Training Mismatch in
  Modern RL Systems"); the lessons cite the blog URL.

## 6. Colab / H100 pilot notebook: Module 12 commands (not run in this build)

From the repo root with `LAB_TARGET=solution`. Record for each: wall time, GPU type, peak memory
(`max_mem_gb` in the metrics or `nvidia-smi`), tokens generated per second (from `seconds` and `len` in
`metrics.jsonl`), and the printed tables. First run the CPU smoke tests of the same code paths
(`python -m frontierlab.posttrain.hf --smoke --run runs/m12/hf-smoke --steps 2`,
`python labs/module-12/lesson-01/rm_main.py --smoke --out runs/m12/l121-smoke`,
`python labs/module-12/lesson-04/eval_main.py --smoke --out runs/m12/l124-smoke`).

| Lesson | Command | PROJECTED cost | Pilot question |
|---|---|---|---|
| 12.1 | `python labs/module-12/lesson-01/rm_main.py --out runs/m12/l121-main` (A100 pilot: add `--gold 1.7b` if the 8B gold model does not fit) | 2–3 GPU-h (generation dominates: 1.6e7 tokens) | does the smallest proxy over-optimise within n = 64? measured generation tok/s |
| 12.2 | `python labs/module-12/lesson-02/estimator_lab.py --variant main --print` → 8 runs of `python -m frontierlab.posttrain.hf ... --steps 100` | 8–12 GPU-h (30–50 s/step) | measured s/step and memory for Qwen3-1.7B-Base at P=32, G=8, 512 new tokens; do estimators separate beyond seed noise? |
| 12.3 | `python labs/module-12/lesson-03/details_lab.py --variant main --print` → 12 runs | 12–18 GPU-h | does len_wrong grow under seq_mean_token_mean at 256+ tokens? truncation rates per arm |
| 12.4 | `python labs/module-12/lesson-04/eval_lab.py --variant main --print` (2 RL runs + `eval_main.py`) | 5–7 GPU-h | Eval v2 cost per checkpoint; IFEval-subset agreement with `lm_eval --tasks ifeval` on the 215 shared prompts |
| project | 2 arms × 2 seeds × 200 steps of `frontierlab.posttrain.hf` + 5 × `eval_main.py` | 9–14 GPU-h | — |

Scaled pilot per plan section 12.1 (row "12–14 RL loop"): Qwen3-0.6B-Base, `--max-new 256`, 2 arms
(`--scale group` vs `--scale none`) × 2 seeds on an A100, 100 steps each; project the 1.7B cost from the
measured s/step as (1.72/0.60) × measured training time + measured generation time × (1.7B decode time /
0.6B decode time), and label it PROJECTED.

## 7. BUILD_PROGRESS / TODO_FOR_TAL

- Stage D base model choice (section 3 above) for Tal to confirm.
- The free-CPU timings in Module 12 were measured with Module 11's jobs sharing the CPU.
