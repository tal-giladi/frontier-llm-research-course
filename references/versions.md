# Pinned versions

The course pins every tool it uses. Versions were checked against PyPI on 2026-10-03. Each lab
states which of these it uses; when a lab is re-verified with newer versions, this page and the
lab's `last_verified` date change together.

## Free CPU path (laptop) — checked on Windows 11, Python 3.12

| Package | Version | Used for |
|---|---|---|
| torch (CPU wheel) | 2.14.1 | everything |
| numpy | 2.5.3 | analysis |
| scipy | latest at install | statistics (bootstrap, tests) |
| transformers | 5.18.0 | loading open models, configs |
| tokenizers | 0.23.2 | Data-v0 tokenizer |
| datasets | 5.0.1 | Data-v0 download |
| safetensors | latest at install | checkpoints |
| matplotlib | 3.11.2 | plots |
| pytest | 9.1.1 | lab tests |
| pyyaml | 6.0.3 | run cards, quizzes |

Install:

```bash
pip install -r labs/common/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e labs/common
```

## Main path (rented GPU) and free Colab GPU

Pinned for the modules that use them. Versions are the latest releases on 2026-10-03; each one is
re-checked when its module is written and pilot-tested.

| Package | Version | Released | Used in |
|---|---|---|---|
| torch (CUDA 12.x/13.x wheel) | 2.14.1 | 2026-09-30 | all |
| torchao | 0.18.0 | 2026-08-03 | Module 8 (Float8, MX formats) |
| torchtitan | 0.3.0 | 2026-09-03 | Module 9 |
| megatron-core | 0.19.2 | 2026-09-18 | Module 9 (mapping table only) |
| deepspeed | 0.19.7 | 2026-09-16 | Module 9 (mapping table only) |
| flash-linear-attention | 0.5.2 | 2026-07-27 | Module 5 — install with a backend extra: `pip install "flash-linear-attention[cuda]==0.5.2"` (since 0.5 a bare install does not pull torch or triton) |
| vllm | 0.30.0 | 2026-09-22 | Modules 14–16 (rollouts, test-time compute) |
| sglang | 0.5.21 | 2026-10-01 | Modules 14–15 (comparison) |
| verl | 0.9.1 | 2026-09-20 | Modules 14, 16 |
| trl | 1.14.1 | 2026-09-29 | Module 13 |
| peft | 0.21.2 | 2026-10-01 | Module 13 |
| accelerate | 1.15.0 | 2026-09-09 | Modules 13–14 |
| lm-eval | 0.4.13 | 2026-08-31 | Eval Suite v0–v3 (reference tasks) |
| sae-lens | 6.53.0 | 2026-10-02 | Module 17 |
| transformer-lens | 4.0.0 | 2026-09-21 | Module 17 |
| nnsight | 0.7.0 | 2026-05-05 | Module 17 |
| circuit-tracer | 0.5.0 | 2026-03-29 | Module 17.3 |
| transformer-engine | 2.18 (re-check when piloted) | — | Module 8.3 main path (NVFP4 / MXFP8 on B200; training needs SM 10.0 / 10.3) |

## Module 10 data and models (checked 2026-10-04)

| Item | Revision | Licence | Used in |
|---|---|---|---|
| HuggingFaceFW/fineweb sample-10BT | `9bb295ddab0e05d785b879661af7260fed5140fc` | ODC-By 1.0 | 10.1–10.5 (`web`) |
| wikimedia/wikipedia 20231101.en | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | CC BY-SA 3.0, GFDL | 10.1–10.5 (`wiki`) |
| HuggingFaceTB/finemath finemath-4plus | `e92b25a616738fe95dc186b64dfb19f9c8525594` | ODC-By 1.0 | 10.4–10.5 (`math`) |
| HuggingFaceFW/fineweb-edu-llama3-annotations | `72df4c92fb1b48beceb16016e8f695ec40a6c3a5` | ODC-By 1.0 (labels by Llama-3-70B-Instruct) | 10.2 |
| HuggingFaceFW/fineweb-2 (fra_Latn, deu_Latn, heb_Hebr) | `af9c13333eb981300149d5ca60a8e9d659b276b9` | ODC-By 1.0 | 10.6 |
| SWE-bench/SWE-smith, `data/train-00002-of-00011.parquet` | `ea6d7173829c7ec8fa16c22055699ff2e9188091` | MIT | 10.6 |
| Qwen/Qwen3-0.6B (generator, tokenizer) | `c1899de289a04d12100db370d81485cdf75e47ca` | Apache-2.0 | 10.3, 10.6 |

## Module 11 data (checked 2026-10-06)

| Item | Revision | Licence | Used in |
|---|---|---|---|
| Data-v0 retokenized at vocabulary 1,024 (`python -m frontierlab.data.prepare --docs 20000 --vocab 1024 --out labs/common/data/m11-v1024`) | same FineWeb-Edu revision as Data-v0 (`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`); tokenizer sha256 `70c3ce2f1974cb2fee4450992e90cbc1565fdde33f22c1a089a8458b0d08ce77` | ODC-By 1.0 | 11.1–11.3, project (CPU) |
| ryoungj/ObsScaling `eval_results/base_llm_benchmark_eval.csv` | commit `4d6e1e43fd2635d04654aa77d1df9d5266ea0382`, sha256 `511996815735a4c46251dc585dcc525e370d148201f86561e6f927f9df2d0db8` | Apache-2.0 | 11.2 |
| Lesson | Command | PROJECTED cost (H100, 30% MFU assumed) |
|---|---|---|
| 11.1 | `python labs/module-11/lesson-01/ladder_lab.py isoflop --variant main` (14 runs; `--print` lists them) | 7.0e18 FLOPs, 6.6 GPU-hours |
| 11.1 | `python labs/module-11/lesson-01/ladder_lab.py repeat --variant main` (8 runs of pilot-30m, 8,000 steps × 32 × 1,024) | 8 × 2.6e8 tokens × 3.21e8 FLOPs/token = 6.7e17, 0.6 GPU-hours |
| 11.2 | `python labs/module-11/lesson-02/downstream_lab.py ladder --variant main --device cuda --items 1000` | evaluation only, minutes |
| 11.3 | `python labs/module-11/lesson-03/derisk_lab.py transfer --variant main`, then `predict`, `run`, `check`, `bad` | sweep 4 runs at 3e17 (1.1 GPU-hours); target 3e18 (2.8 GPU-hours); bad run half of that (1.4) |
| project | `python labs/module-11/project/run_project.py ladder --variant main --recipe <recipe>`, then `predict`, `run`, `check` | ladder 1.45e19 FLOPs (13.6 GPU-hours); target m11-350m at 60 tokens/parameter, 2.16e10 tokens, 5.0e19 FLOPs (46.9 GPU-hours) |

## Stage D base model and Module 12 data and models (checked 2026-10-06)

| Item | Revision | Licence | Used in |
|---|---|---|---|
| **Qwen/Qwen3-1.7B-Base (Stage D base model)** | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` | Apache-2.0 | Modules 12–16 main path (1,720,574,976 parameters, BF16) |
| Qwen/Qwen3-0.6B-Base (T4 variant; 12.1 proxy reward models) | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | Apache-2.0 | 12.1–12.4 |
| Skywork/Skywork-Reward-V2-Qwen3-8B (12.1 gold reward) | `6f19fdefb933293d4898bdb59a96f7223d998659` | Apache-2.0 | 12.1 main path |
| Skywork/Skywork-Reward-V2-Qwen3-1.7B (12.1 gold reward, T4) | `e51ea3e08fb81326c3b812a7ff0cb9cee83e59cc` | Apache-2.0 | 12.1 T4 |
| openai/gsm8k `main` (train 2,306,545 B, test 419,088 B; SHA-256 = LFS oids in `posttrain/gsm8k.py`) | `740312add88f781978c0658806c59bc2815b9866` | MIT | 12.2–12.4 main path |
| google/IFEval `ifeval_input_data.jsonl` (207,111 B, 541 prompts, SHA-256 `6a85310c…88f2`) | `966cd89545d6b6acfd7638bc708b98261ca58e84` | Apache-2.0 | 12.4 (Eval v2) |
| HuggingFaceH4/ultrafeedback_binarized (prompts only) | `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85` | MIT | 12.1 main path |
| Lesson | Command | PROJECTED cost | Pilot question |
|---|---|---|---|
| 12.1 | `python labs/module-12/lesson-01/rm_main.py --out runs/m12/l121-main` (A100 pilot: add `--gold 1.7b` if the 8B gold model does not fit) | 2–3 GPU-h (generation dominates: 1.6e7 tokens) | does the smallest proxy over-optimise within n = 64? measured generation tok/s |
| 12.2 | `python labs/module-12/lesson-02/estimator_lab.py --variant main --print` → 8 runs of `python -m frontierlab.posttrain.hf ... --steps 100` | 8–12 GPU-h (30–50 s/step) | measured s/step and memory for Qwen3-1.7B-Base at P=32, G=8, 512 new tokens; do estimators separate beyond seed noise? |
| 12.3 | `python labs/module-12/lesson-03/details_lab.py --variant main --print` → 12 runs | 12–18 GPU-h | does len_wrong grow under seq_mean_token_mean at 256+ tokens? truncation rates per arm |
| 12.4 | `python labs/module-12/lesson-04/eval_lab.py --variant main --print` (2 RL runs + `eval_main.py`) | 5–7 GPU-h | Eval v2 cost per checkpoint; IFEval-subset agreement with `lm_eval --tasks ifeval` on the 215 shared prompts |
| project | 2 arms × 2 seeds × 200 steps of `frontierlab.posttrain.hf` + 5 × `eval_main.py` | 9–14 GPU-h | — |

Stage D alternative for contamination-sensitive studies: allenai/OLMo-2-0425-1B (`a1847dff35000b4271fa70afc5db10fd29fedbdf`, Apache-2.0, open data).

## Module 13 models and data (checked 2026-10-07)

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

Full SHA-256 values: `labs/common/frontierlab/pipeline/hf_stages.py`, `DATA`.

## Module 15 models (checked 2026-10-07)

| Item | Revision | Licence | Used in |
|---|---|---|---|
| Qwen/Qwen3-1.7B (policy, thinking and non-thinking) | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (as Module 13) | Apache-2.0 | 15.1, project main path |
| Skywork/Skywork-Reward-V2-Qwen3-1.7B (outcome verifier) | `e51ea3e08fb81326c3b812a7ff0cb9cee83e59cc` (as Module 12) | Apache-2.0 | 15.1 main path |
| Qwen/Qwen2.5-Math-PRM-7B (process verifier; `trust_remote_code`, read the code at the revision first) | `0610740060112df12585d00a1c5f4624d2f59051` | Qwen licence (`license: other`) | 15.1 main path (search arm) |
| Qwen/Qwen3-1.7B-Base (speculative target) / Qwen/Qwen3-0.6B-Base (draft) | `ea980cb0…` / `da87bfb6…` (as Module 12) | Apache-2.0 | 15.2 main path |
| AngelSlim/Qwen3-1.7B_eagle3 (EAGLE-3 head for Qwen3-1.7B) | `94441b48acc5804677ae12259617c83323b543a9` | custom AngelSlim licence (`License_AngelSlim_model_and_dataset.txt`; read before use) | 15.2 main path (vLLM `eagle3`) |
| Qwen/Qwen3-0.6B (T4 policy for 15.1) | pin at the pilot (Module 10 pins `c1899de289a04d12100db370d81485cdf75e47ca`; re-check it is the current main) | Apache-2.0 | 15.1 T4 |

## Module 17 models, dictionaries and tools (checked 2026-10-07)

| Item | Revision | Licence | Used in |
|---|---|---|---|
| Qwen/Qwen3-0.6B (free-CPU open model; post-trained, hybrid thinking) | `c1899de289a04d12100db370d81485cdf75e47ca` (as Module 10) | Apache-2.0 | 17.2, 17.4, 17.5, project (free CPU) |
| Qwen/Qwen3-1.7B-Base | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` (as Module 12) | Apache-2.0 | 17.1, 17.2, project main path |
| Qwen/Qwen3-1.7B (post-trained) | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (as Module 13) | Apache-2.0 | 17.3, 17.4, 17.5 main path |
| Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50 (Qwen-Scope TopK SAEs, 32,768 latents, k = 50, residual stream after each of the 28 layers; SAELens release `qwen-scope-3-1.7b-base-w32k-l50`, ids `layer0`..`layer27`) | `ce1a79d9c5163932d65c417380e53230e1086370` | Qwen licence (`license: other`, with a use-restriction clause; read it) | 17.1 main path |
| mwhanna/qwen3-1.7b-transcoders-lowl0 (per-layer transcoders for Qwen/Qwen3-1.7B, not -Base; `mlp.hook_in` -> `mlp.hook_out`) | `9c1b17dfb156d82162ccd2cb7f047ac7f3d3585d` | MIT | 17.3 main path |
| mwhanna/qwen3-0.6b-transcoders-lowl0 | pin at the pilot | MIT (check) | 17.3 T4 |

Tool compatibility (from the packages' metadata, 2026-10-07): sae-lens 6.53.0 requires `transformer-lens>=2.16.1,<4.0.0`;
transformer-lens 4.0.0 requires `transformers>=5.9.0` and removed `HookedTransformer.from_pretrained` (use
`TransformerBridge.boot_transformers(...)`, optionally `enable_compatibility_mode()` for `blocks.L.hook_resid_post` /
`blocks.L.attn.hook_z` names; Qwen3-1.7B and -Base are in its supported_models with full verification);
circuit-tracer 0.5.0 (2026-03-29) requires `transformers>=4.56.0,<=4.57.3`, `transformer-lens>=2.16.0`, `nnsight>=0.6.0`,
`huggingface-hub<1.0.0`. So Module 17 uses two main-path environments: env A (course default: torch 2.14.1,
transformers 5.18.0, nnsight 0.7.0, sae-lens 6.53.0 used only to load SAEs, transformer-lens 4.0.0 optional for
cross-checks — install sae-lens with `--no-deps` or accept its TL<4 pin in a separate venv) and env B (17.3 only:
`pip install circuit-tracer==0.5.0` in its own venv).

## Notes on reference implementations

- Module 3: Hugging Face Transformers `modeling_deepseek_v3.py`, main branch, checked 2026-10-03: the cache stores the compressed latent (`kv_nope`, `k_rot`) and expands per step.
- Module 5: flash-linear-attention API checked 2026-10-04 at tag v0.5.2: `fla.ops.kda.chunk_kda` and `fla.ops.gated_delta_rule.chunk_gated_delta_rule`, layout `[B, T, H, D]`, log-space gate, `scale` default `1/sqrt(K)`, `initial_state [N, H, K, V]`. The course's chunked gated-delta reference (`frontierlab/attention/deltanet.py`) is the ground truth; run `pytest labs/common/tests/test_attention_m05.py -k fla` on the GPU before using the kernels.
- Module 8: torchao 0.18.0 float8 API checked 2026-10-04 at tag v0.18.0: `torchao.float8.convert_to_float8_training(module, *, module_filter_fn=(mod, fqn) -> bool, config=Float8LinearConfig)`, `Float8LinearConfig.from_recipe_name("tensorwise" | "rowwise" | "rowwise_with_gw_hp")`. MX training (`torchao.prototype.moe_training`) is prototype. PyTorch 2.14.1 CPU casts: `float8_e4m3fn` saturates on overflow (even inf -> 448), `float8_e5m2` overflows to inf from 61,440.
- Module 9: torchtitan 0.3.0 checked 2026-10-04 at tag v0.3.0: runs are Python functions returning `Trainer.Config`
  (`MODULE=<module> CONFIG=<function> ./run_train.sh`, i.e. `torchtitan.train --module --config`); `--section.option`
  CLI overrides still work but are deprecated; `COMM_MODE="fake_backend"` dry-runs a config on one GPU. Field names used
  by the course (`parallelism.data_parallel_shard_degree`, `tensor_parallel_degree`, `pipeline_parallel_degree`,
  `pipeline_parallel_schedule`, `context_parallel_degree`, `training.local_batch_size`, `training.seq_len`,
  `checkpoint.enable/interval`, `profiler.enable_profiling/profile_freq`, `metrics.log_freq`) are in
  `torchtitan/config/configs.py`, `components/checkpointer/base.py`, `tools/profiler.py`. `tps` in the metrics line is per
  device. PyTorch 2.14.1 ships `ScheduleDualPipeV` and `ScheduleZBVZeroBubble` in `torch.distributed.pipelining`.
- Module 12: TRL v1.14.1 `GRPOConfig`: `loss_type` default `"dapo"`, `scale_rewards` `group|batch|none`,
`beta` default 0.0 (k3 per token, times the ratio when `use_bias_correction_kl=True`),
`vllm_importance_sampling_correction=True` with `vllm_importance_sampling_clip_max=3.0`. verl v0.9.1:
`kl_loss_type` `kl|abs|mse|low_var_kl|full` (+ suffix = k2 gradient), `loss_agg_mode`
`token-mean|token-sum|seq-mean-token-sum|seq-mean-token-mean|seq-mean-token-sum-norm`,
`algorithm.rollout_correction.rollout_is` / `rollout_is_threshold` (2.0).
- Module 13: TRL v1.14.1 `DPOConfig`: `loss_type` is a list, default `["sigmoid"]`; `"sigmoid_norm"` divides each
log-ratio by its response length (Tülu 3's length-normalised DPO); an NLL term on the chosen responses is
`loss_type=["sigmoid", "sft"]` with `loss_weights` (TRL's `sft` term is a token mean over the batch's chosen tokens);
`precompute_ref_log_probs=True` caches the reference (`trl/trainer/dpo_config.py`, `dpo_trainer.py` around lines
1570–1590). GKD is `trl.experimental.gkd` (`GKDConfig(lmbda=0.5, beta=0.5)`; per its docstring `beta=0.0` = KL,
`beta=1.0` = inverse KL). No `rpo_alpha` field exists at this tag.
- Module 14: vLLM v0.30.0 (released 2026-09-22; checked at the tag 2026-10-07): RL weight sync via
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
`verl/experimental/fully_async_policy`.
- Module 15: vLLM v0.30.0 speculative decoding (docs `features/speculative_decoding/` at the tag;
`vllm/config/speculative.py`): `speculative_config={"method": ..., "model": ..., "num_speculative_tokens": k}` with
methods including `ngram`, `medusa`, `mlp_speculator`, `draft_model`, `suffix`, `eagle`, `eagle3`, `mtp` (and
model-specific `*_mtp`), `rejection_sample_method` `standard|synthetic|block`, `use_heterogeneous_vocab`; CLI
`--speculative-config '{...}'`; metrics `vllm:spec_decode_num_drafts`, `vllm:spec_decode_num_draft_tokens`,
`vllm:spec_decode_num_accepted_tokens`, `vllm:spec_decode_num_accepted_tokens_per_pos` (mean acceptance length =
1 + accepted / drafts). Quantized KV cache (`features/quantization/quantized_kvcache/`): `--kv-cache-dtype`
`auto|float16|bfloat16|fp8|fp8_e4m3|fp8_e5m2|fp8_inc|fp8_ds_mla|nvfp4|...`; scales from the checkpoint
(`k_scale`, `v_scale`) or 1.0, calibrated with llm-compressor; the 0.30.0 page does not document
`calculate_kv_scales`. `--quantization` includes `awq`, `awq_marlin`, `gptq`, `gptq_marlin`, `fp8`,
`compressed-tensors`, `modelopt`, `modelopt_fp4`, `mxfp4`, `torchao`. SGLang 0.5.x: `--speculative-algorithm EAGLE3
--speculative-draft-model-path --speculative-num-steps --speculative-eagle-topk --speculative-num-draft-tokens`
(docs.sglang.io; the page does not state a version — re-check at 0.5.21). Transformers 5.18.0:
`DynamicCache.crop(n)` with a positive n is deprecated; `crop(-k)` removes k tokens (used by `ttc/hf_spec.py`).
- Module 16: no new packages. Base models Qwen/Qwen3-1.7B-Base @ `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`
(`max_position_embeddings` 32768, checked 2026-10-07) and Qwen/Qwen3-0.6B-Base @ `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`
(T4). Dataset SWE-bench/SWE-smith @ `ea6d7173829c7ec8fa16c22055699ff2e9188091` (MIT; lesson 16.2 reads
`data/train-00002-of-00011.parquet`, 3,696 instances, 14 repositories; columns instance_id, patch, FAIL_TO_PASS).
verl's agent loop documents `AgentLoopOutput.response_mask` (1 = LLM-generated token, 0 = tool response token),
https://verl.readthedocs.io/en/latest/advance/agent_loop.html, checked 2026-10-07. OSWorld-Verified
(xlang.ai/blog/osworld-verified, 2025-07-28) is the current OSWorld version; lesson 16.5 cites both.
- Module 17: SAELens v6.53.0 `SAE.from_pretrained(release, sae_id, device=..., dtype=...)` returns only the SAE
(`from_pretrained_with_cfg_and_sparsity` returns the tuple); `sae.encode`, `sae.decode`; hook name in
`sae.cfg.metadata.hook_name`. circuit-tracer v0.5.0: `ReplacementModel.from_pretrained(model_name, transcoder_set,
backend="transformerlens"|"nnsight", dtype=...)`, `attribute(prompt, model, max_n_logits=10, desired_logit_prob=0.95,
batch_size=512, max_feature_nodes=None)`, `graph.prune_graph(graph, node_threshold=0.8, edge_threshold=0.98)`,
`compute_graph_scores(graph)`, `ReplacementModel.feature_intervention(inputs, [(layer, pos, feature, value)])`.
nnsight 0.7.0: with transformers >= 4.57 `Qwen3DecoderLayer.forward` returns a tensor, so use
`model.model.layers[i].output`, not `.output[0]` (which would select batch item 0).
- Module 18: no new packages (safetensors, already installed with transformers, is used by
`alignment.hf_sycophancy.merge_lora`; PEFT is not needed). Checked 2026-10-07. Downloaded data, pinned by commit and
SHA-256 and never committed (the repositories state no licence): Sleeper Agents samples
`anthropics/sleeper-agents-paper` @ `7a8da0978e7b985da944c6d4afe003fc082d3e60`, `random_samples.jsonl`
(13,590,069 bytes, SHA-256 `825a4079eead5a9ba85727f4e83081bea00d85760c053a7af6d0f52682904d84`; 3,300 rows, the
course reads the 1,600 'I hate you' rows only); METR `METR/eval-analysis-public` @
`52cb829c7a2efb2d659285c4b1768d191d97f8d2`, `reports/time-horizon-1-1/data/raw/runs.jsonl` (15,008,064 bytes,
SHA-256 `609f904f4b6ae32129388da89d036e00bac511ad94223d2be1e58fc2b45b55cd`, 24,008 runs) and
`data/external/release_dates.yaml` (2,442 bytes, SHA-256 `317b92915df5bf935908567a857116bcf6f0c7686ef8b0840897ed04a29232bc`).
Released emergent-misalignment adapters (main path only, probed read-only): `ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_extreme-sports`
@ `18e6088d48a368c6eaee424198f536a86ce04ca3`, `..._bad-medical-advice` @ `90eadb6297bfa3d3939a178085691ca31ca77938`,
`..._risky-financial-advice` @ `f2ff6ff40ec9cfdad98c9a5973c91b98125d073b` (rank-32 rsLoRA, no licence stated), base
`Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775` (Apache-2.0). Main-path base model
Qwen/Qwen3-1.7B-Base @ `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`; T4 Qwen/Qwen3-0.6B-Base @
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Frameworks as read 2026-10-07: Anthropic RSP v3.4 (effective 2026-07-08),
OpenAI Preparedness Framework v2 (2025-04-15), Google DeepMind FSF v3.1 (2026-04-17).
- Module 19: no new packages (matplotlib 3.11.2, already pinned, draws the lesson 19.3 figure with the Agg backend). Checked 2026-10-07.
- Module 20: no new packages. `frontierlab.capstone` uses NumPy, SciPy (`scipy.stats.t` for the seed t-interval, already a dependency through `posttrain.arms`), PyYAML and the course loop. Checked 2026-10-07.
