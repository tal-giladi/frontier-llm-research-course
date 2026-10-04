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
