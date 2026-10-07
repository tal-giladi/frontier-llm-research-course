# Module 14 — proposed changes to shared files (for the main session)

Module 14 edits no existing `frontierlab` file, `_sidebar.md`, `glossary.md`, `references/` or `templates/`.
New code: `labs/common/frontierlab/rlscale/` (objectives, runner, stability, asyncsim, mismatch, curves, passk,
hf_rl, vllm_rollout), tests `labs/common/tests/test_rlscale.py`. It imports `frontierlab.posttrain` and
`frontierlab.evals.suite_v2` and wraps the Module 12 loop at run time (`rlscale.runner.patched_loop`, restored on exit).

## 1. `_sidebar.md`: Module 14 lines

Running numbers 59–62 (fix them if Module 13 ends elsewhere):

```markdown
- **Module 14 — Which RL objective, at what scale?**
  - [59 · Objectives derived, not switched](lessons/module-14/lesson-01.md)
  - [60 · A controlled objective comparison](lessons/module-14/lesson-02.md)
  - [61 · Rollout systems and staleness](lessons/module-14/lesson-03.md)
  - [62 · RL scaling and the capability debate](lessons/module-14/lesson-04.md)
  - [Module 14 quiz](assessments/module-14-quiz.md)
```

The project page `projects/module-14-reasoning-rl.md` is picked up from `projects/` automatically.

## 2. `glossary.md`

Merge `curriculum/glossary-inbox/module-14.md` (34 terms).

## 3. `references/versions.md`

Add under "Notes on reference implementations":

"Module 14: vLLM v0.30.0 (released 2026-09-22; checked at the tag 2026-10-07): RL weight sync via
`vllm.distributed.weight_transfer` (`WeightTransferTrainerFactory.trainer_init(init_info=IPCTrainerInitInfo(rank=0,
packed=False), client=HTTPVLLMWeightSyncClient(url), source=ModuleSource(model))`, `send_weights()`), server flag
`--weight-transfer-config '{"backend": "ipc"|"nccl"|...}'`, `VLLM_SERVER_DEV_MODE=1` for `POST /pause`, `/resume`;
`VLLM_ALLOW_INSECURE_SERIALIZATION=1` for IPC handles (`examples/rl/rlhf_http_ipc.py`); batch-invariant mode
`VLLM_BATCH_INVARIANT=1` (beta; `docs/features/batch_invariance.md`; docs say compute capability >= 8.0, the
`envs.py` comment says >= 9.0 — check on the pilot GPU); sleep mode `LLM(enable_sleep_mode=True)`, `sleep(level)`,
`wake_up(tags)`. vLLM v0.31.0 was released 2026-10-05; the course stays on 0.30.0 until re-verified.
verl v0.9.1: `actor_rollout_ref.actor.policy_loss.loss_mode` in {vanilla, dppo_tv, dppo_kl, gspo, sapo, gpg, clip_cov,
kl_cov, geo_mean, dro, cispo, bypass_mode} (`verl/trainer/ppo/core_algos.py`), `clip_ratio_low/high` 0.2,
`clip_ratio_c` 3.0; verl's `cispo` clamps with `clip_ratio_low/high`; async recipes `verl/experimental/one_step_off_policy`,
`verl/experimental/fully_async_policy`."

The main-path vLLM path needs `requests` and `openai`-free HTTP only (`requests`), installed with vLLM on the GPU
machine. No new CPU requirements.

## 4. Proposed hooks for `frontierlab/posttrain/rl.py` (optional; Module 14 works without them)

Module 14 currently wraps the loop at run time: it registers control verifiers in `tasks.VERIFIERS`, replaces
`rl.evaluate` with a strict-verifier wrapper, and replaces `rl.Sampler` with a bounded-staleness subclass, all
inside `rlscale.runner.patched_loop` (restored in `finally`). A cleaner loop would expose:

1. `RLConfig.eval_verifier: str = "strict"` used by `evaluate(...)` instead of `cfg.verifier` (so a control reward
   never changes evaluation), and log the strict training pass rate next to the training reward (`pass_strict`).
2. `hooks["sampler"]`: a factory `(policy, cfg) -> Sampler` used instead of `Sampler(policy, cfg.staleness, cfg.sampler_dtype)`.
3. `hooks["reward"]`: `(rollout, problems, cfg) -> rewards (B,)` used instead of `score(...)` for training only.

If these are added, `rlscale.runner.train_objective` can pass them instead of patching (a 20-line change there).

## 5. Plan section 14.1 claim checks to record

- V: GRPO DeepSeekMath Eq. 3 (KL inside the per-token sum, k3 estimator Eq. 4); section 4.2: lr 1e-6, beta 0.04,
  G = 64, max length 1024, batch 1024; epsilon not stated.
- V: DAPO objective is Eq. 8 (Eq. 11 dynamic-sampling constraint alone, Eq. 12 token-level form); G = 16, prompt
  batch 512, lr 1e-6 (section 4.1). (Lesson 12.3 cites Eq. 12 for the token-level loss — both are correct.)
- V: GSPO Eq. 5 (objective), Eq. 7 (s_i), gradients Eqs. 8–10 and 11–12, GSPO-token Eqs. 13–14 (arXiv HTML v2;
  numbering differs between renderings, so lessons cite sections). Clipped-token fraction "two orders of
  magnitude" higher than GRPO is qualitative in the text (numbers only in Fig. 2). GSPO "contributed to the
  remarkable improvements in the latest Qwen3 models".
- V: CISPO (MiniMax-M1 section 3.1): no lower bound (eps_low^IS large), only eps_high^IS tuned, its value not
  stated; 2x speed-up over DAPO, matching it with 50% of steps (Fig. 2); fork tokens "However, Recheck, Wait, Aha".
  FP32 LM head raised train/inference probability correlation from ~0.9x to 0.99x (section 3.2).
- V: ScaleRL: fit windows for the 8B 100k run: (1.5k, 50k) A = 0.645, B = 1.70; (0, 100k) A = 0.655, B = 1.56;
  run-to-run A within ±0.015 (3 runs), ±0.02 margin; FP32 logits A 0.52 -> 0.61 (Fig. 5b); CISPO B 2.01 vs DAPO
  1.77 (Fig. 7); 100k GPU-hours 8B, 50k Scout (Fig. 1); validation 1,000 Polaris-53k prompts, 16 generations, every
  100 steps. **Not found in text: C_mid values; A values per loss type in Fig. 5a (figure only).** Lesson 14.4 uses
  C_mid = 8,000 and R_0 = 0.30 as labelled stand-ins.
- V: Yue et al.: largest k 1,024 (AIME24, AMC23), 128 (MATH500, GSM8K, Minerva), 256 (code, visual); estimator Eq. 2.
- V: Thinking Machines: Qwen3-235B-A22B-Instruct-2507, 1,000 completions, 80 unique; 26 s default vs 42 s with the
  improved deterministic attention (55 s unoptimised); RLVR without IS correction collapses vs bitwise-identical
  sampler/trainer.
- V: AReaL bound floor((N_r - 1)/B) <= i + eta (section 5.1), eta 4 (code) / 8 (math), decoupled PPO Eq. 5, up to
  2.77x (abstract), Table 1 2.57x (1.5B) and 2.24x (14B).
- V: PipelineRL arXiv 2509.19128: in-flight weight updates without KV recompute (section 4), ~2x on 128 H100.
- V: Asynchronous RLHF: online DPO most robust (section 3.3), ~40% faster LLaMA 3.1 8B (section 5.1).
- V: Spurious rewards: random +21.4, incorrect +24.1, ground truth +29.1 MATH-500 on Qwen2.5-Math-7B; format +13.8
  (stated in an "AMC" sentence whose numbers match MATH-500 — the lessons do not use the format number).
- V: Qwen3 report section 4.2: reasoning RL with GRPO on 3,995 query-verifier pairs.
- V: ProRL (arXiv 2505.24864): KL control, reference policy resetting; claims pass@k gains including where the base
  fails entirely.
- V: Qwen3-1.7B-Base config at `ea980cb`: 28 layers, 16 query / 8 KV heads, head_dim 128 (KV 114,688 B/token in bf16).

## 6. Colab / H100 pilot notebook: Module 14 commands (not run in this build)

From the repo root with `LAB_TARGET=solution`. First the CPU smoke test of the same code path:
`python -m frontierlab.rlscale.hf_rl --smoke --run runs/m14/hf-smoke --steps 2 --objective cispo`.
Record for each run: GPU type, s/step (`seconds` in `metrics.jsonl`), peak memory (`max_mem_gb`), tokens generated
per second, `mismatch_*` columns (vLLM runs), and the printed tables. Per-step cost basis: lesson 12.2's 30–50 s/step
for Qwen3-1.7B-Base at P = 32, G = 8, 512 new tokens with transformers `generate` (itself PROJECTED).

| Lesson | Command | PROJECTED cost (H100) | Pilot question |
|---|---|---|---|
| 14.1 | `python labs/module-14/lesson-01/objectives_lab.py --variant main --print` → 5 × 50 steps | 2.1–3.5 GPU-h | do clip fractions at 512 tokens follow the 14.1 derivations (GSPO ≫ GRPO; CISPO capped share)? |
| 14.2 | `python labs/module-14/lesson-02/compare_lab.py --variant main --print` → 15 tuning + 12 comparison runs, then `eval_main.py` per final policy | 31–52 GPU-h | do any two objectives separate beyond 2-seed intervals at 1.7B? does the random control gain on Qwen3-1.7B-Base GSM8K? |
| 14.3 | `python labs/module-14/lesson-03/async_lab.py --variant main --print` → 5 × 100 steps (HF; vLLM sync; vLLM sync batch-invariant; vLLM k=1 GRPO; vLLM k=1 CISPO) | 2.8–7 GPU-h | measured generation share of step time; trainer/vLLM mismatch with and without `VLLM_BATCH_INVARIANT=1`; speed-up of k=1; does the vLLM `logprobs` field match the processed distribution? |
| 14.4 | `python labs/module-14/lesson-04/scaling_lab.py --variant main --print` → 2 × 300 steps with 64 samples × 200 questions every 25 steps | 12–21 GPU-h | is the asymptote of a 300-step run bounded? does a pass@k crossover appear on GSM8K by k = 64? |
| project | `python labs/module-14/project/run_project.py --variant main --objective <chosen> --print` → 4 runs × 300 steps + 4 × `eval_main.py` | 25–47 GPU-h | — |

Scaled pilot per plan section 12.1 (row "12–14 RL loop and objective comparison"): Qwen3-0.6B-Base on an A100,
`--max-new 256 --prompts 16`, GRPO vs CISPO × 2 seeds × 100 steps, plus the vLLM sync/k=1 pair; project 1.7B cost as
(1.72/0.60) × measured training time + measured generation time × (1.7B decode time / 0.6B decode time), labelled
PROJECTED. The 14.4 long-run fit stays "not produced course-side" (published curves only), as the plan says.

## 7. BUILD_PROGRESS / TODO_FOR_TAL

- Free-CPU timings in Module 14 were measured with Module 13's jobs and a second Module 14 script on the same CPU.
- Lesson 14.4's reconstructed ScaleRL curve uses stand-in C_mid and R_0 (not in the paper's text); a digitised
  Figure 1 would replace them (a pilot-notebook task, a few minutes with WebPlotDigitizer).
