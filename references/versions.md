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
