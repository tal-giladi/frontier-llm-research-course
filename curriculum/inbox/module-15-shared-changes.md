# Module 15 — proposed changes to shared files (for the main session)

Module 15 edits no existing `frontierlab` file, `_sidebar.md`, `glossary.md`, `references/`, `templates/` or plan
file. New code: `labs/common/frontierlab/ttc/` (select, budget, world, report, speculative, eagle, spec_models,
serving, kvquant, hf_ttc, hf_spec), tests `labs/common/tests/test_ttc.py` (27 tests, about 70 s). It imports
`frontierlab.perf`, `attention.accounting`, `blocks.mtp`, `blocks.train`, `posttrain`, `pipeline`, `evals.suite_v2`,
`precision`, `calc`, `stats` and never patches them.

## 1. `_sidebar.md`: Module 15 lines

Running numbers 63–66 (after Module 14's 59–62; fix them if earlier modules end elsewhere). 15.4 is an extension:

```markdown
- **Module 15 — How should compute be spent at inference?**
  - [63 · A test-time compute experiment](lessons/module-15/lesson-01.md)
  - [64 · Speculative decoding](lessons/module-15/lesson-02.md)
  - [65 · Serving cost of architecture choices](lessons/module-15/lesson-03.md)
  - [66 · KV and weight quantisation for serving](lessons/module-15/lesson-04.md)
  - [Module 15 quiz](assessments/module-15-quiz.md)
```

The project page `projects/module-15-ttc-report.md` is picked up from `projects/` automatically.

## 2. `glossary.md`

Merge `curriculum/glossary-inbox/module-15.md` (38 terms).

## 3. `references/versions.md`

Add a section:

"## Module 15 models (checked 2026-10-07)

| Item | Revision | Licence | Used in |
|---|---|---|---|
| Qwen/Qwen3-1.7B (policy, thinking and non-thinking) | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (as Module 13) | Apache-2.0 | 15.1, project main path |
| Skywork/Skywork-Reward-V2-Qwen3-1.7B (outcome verifier) | `e51ea3e08fb81326c3b812a7ff0cb9cee83e59cc` (as Module 12) | Apache-2.0 | 15.1 main path |
| Qwen/Qwen2.5-Math-PRM-7B (process verifier; `trust_remote_code`, read the code at the revision first) | `0610740060112df12585d00a1c5f4624d2f59051` | Qwen licence (`license: other`) | 15.1 main path (search arm) |
| Qwen/Qwen3-1.7B-Base (speculative target) / Qwen/Qwen3-0.6B-Base (draft) | `ea980cb0…` / `da87bfb6…` (as Module 12) | Apache-2.0 | 15.2 main path |
| AngelSlim/Qwen3-1.7B_eagle3 (EAGLE-3 head for Qwen3-1.7B) | `94441b48acc5804677ae12259617c83323b543a9` | custom AngelSlim licence (`License_AngelSlim_model_and_dataset.txt`; read before use) | 15.2 main path (vLLM `eagle3`) |
| Qwen/Qwen3-0.6B (T4 policy for 15.1) | pin at the pilot (Module 10 pins `c1899de289a04d12100db370d81485cdf75e47ca`; re-check it is the current main) | Apache-2.0 | 15.1 T4 |"

and under "Notes on reference implementations":

"- Module 15: vLLM v0.30.0 speculative decoding (docs `features/speculative_decoding/` at the tag;
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
`DynamicCache.crop(n)` with a positive n is deprecated; `crop(-k)` removes k tokens (used by `ttc/hf_spec.py`)."

No new CPU requirements. The main path needs `vllm==0.30.0` on the GPU machine.

## 4. Optional: attention registry

`frontierlab/ttc/kvquant.py` registers the kind `"gqa-kvq"` when imported (lesson 15.4). If the main session wants it
available from `ModelConfig(attention="gqa-kvq")` without the import, add at the end of `attention/__init__.py`:

```python
import frontierlab.ttc.kvquant  # noqa: F401,E402  (registers "gqa-kvq")
```

(It imports `frontierlab.attention.gqa`, so it must come after the other imports, as `longctx.attention` does.)

## 5. Plan section 14.1 claim checks to record (all checked 2026-10-07 against the primary source)

- V: Snell et al. 2408.03314: ">4x" efficiency vs best-of-N and FLOPs-matched "outperform a 14x larger model" on
  problems with non-trivial success (abstract); PaLM 2-S*, MATH; beam search beats best-of-N at low budgets and
  underperforms at high budgets / easy questions (§5.3); sequential vs parallel ratio by difficulty (§6.2).
  Item 22's scoped form holds.
- V: Brown et al. 2407.21787: SWE-bench Lite 15.9% → 56% with 250 samples (DeepSeek-Coder-V2-Instruct, §2.1);
  exponentiated power law (§3.1); majority vote / RMs plateau "beyond several hundred samples" (abstract) — §4.1 text
  says about 100. Lessons quote the abstract and note the difference.
- V: Self-consistency +17.9% GSM8K (abstract). Cobbe et al.: ~30x size-equivalent (conclusion); performance rises
  to 400 completions then declines (§5.1, Fig. 7a). Lightman et al.: 78.2% on the MATH 500 subset, best-of-1860;
  PRM800K 800K step labels (§3). Math-Shepherd hard/soft estimation (Eqs. 3–4). s1: budget forcing (§3.1), AIME24
  50% → 57% (§4.2), up to 27% over o1-preview. Stroebl et al.: false positives bound resampling; optimal N often < 10.
  REBASE (§3.1.2); Llemma-7B ≈ 34B at ~half the FLOPs (§4.2).
- V: Leviathan et al.: Algorithm 1; Eq. 1 (tokens per round); Def. 3.1–3.2, Thm 3.5, Cor. 3.6 (alpha = E[min(p,q)]
  = 1 − E[TV]); Thm 3.8 (walltime); Table 2 (3.4x/2.6x translation, 3.1x/2.3x summarisation, T5-XXL). Chen et al.:
  2–2.5x Chinchilla 70B. EAGLE: 2.7x–3.5x LLaMA2-Chat 70B; loss L_reg + 0.1 L_cls (§3.2). EAGLE-2: 3.05x–4.26x.
  EAGLE-3: up to 6.5x (Table 1), ~1.4x over EAGLE-2, 1.38x vs SGLang *without* speculation at batch 64 (Table 3).
  Medusa: >2.2x / 2.3–3.6x (latest version; v1 says 2.3–2.8x). DeepSeek-V3 §5.4.3: 85–90%, 1.8x TPS.
- V: DistServe 7.4x requests / 12.6x tighter SLO, >90% within constraints; Splitwise 1.4x at 20% lower cost / 2.35x
  same cost and power; Mooncake up to 525% (simulated) / 75% more requests (real); Sarathi-Serve 2.6x (Mistral-7B,
  1 A100), up to 3.7x (Yi-34B), 5.6x (Falcon-180B); PagedAttention 2–4x.
- V: KIVI G = 32, R = 128 (§4.1), 2.6x peak memory, 4x batch, 2.35x–3.47x throughput; KVQuant <0.1 PPL at 3-bit,
  1M context on one A100-80GB, 10M on 8 GPUs; GPTQ ~4 GPU-hours for 175B at 3–4 bits; AWQ 1% salient weights;
  SmoothQuant W8A8 1.56x, 2x memory.
- V: configs: Qwen3-1.7B-Base `ea980cb` (28 layers, 16/8 heads, head_dim 128, hidden 2048, intermediate 6144, vocab
  151,936, tied); Qwen3-0.6B-Base `da87bfb` (28 layers, 16/8, 128, hidden 1024); DeepSeek-V3 (kv_lora_rank 512,
  qk_rope 64, 61 layers); gpt-oss-120b (36 layers, 18 sliding of 128 + 18 full, 8 KV heads of 64).

## 6. Colab / H100 pilot notebook: Module 15 commands (not run in this build)

From the repo root with `LAB_TARGET=solution`. First the CPU smoke tests of the same code paths:
`python -m frontierlab.ttc.hf_ttc smoke --out runs/m15/hf-smoke` and
`python -m frontierlab.ttc.hf_spec own --smoke --out runs/m15/l152-smoke`. Record for each run: GPU type, wall-clock,
generated tokens/s, the printed tables and JSON files.

| Lesson | Command | PROJECTED cost (H100) | Pilot question |
|---|---|---|---|
| 15.1 | `python -m frontierlab.ttc.hf_ttc sample --out runs/m15/l151-main --n-questions 500 --n 32 --sample-budget 512` | 0.7–1.5 GPU-h (2 × 5.6e6 generated tokens) | measured tokens/s with prefix caching; does thinking-mode sampling at a 512 budget beat non-thinking? |
| 15.1 | `... single --out runs/m15/l151-main --n-questions 500 --budgets 256,512,1024,2048,4096` | 0.3–0.6 GPU-h | accuracy vs thinking budget on Qwen3-1.7B |
| 15.1 | `... score --out runs/m15/l151-main` | 0.2–0.4 GPU-h (32,000 ORM passes) | Skywork ORM AUC and false-positive rate on GSM8K samples; does best-of-N fall with N? |
| 15.1 | `... search --out runs/m15/l151-main --n-questions 200 --width 4 --expand 4` | 0.5–1 GPU-h | does the 7B PRM fit next to vLLM at `gpu_memory_utilization=0.45`? PRM search vs weighted vote at equal pte |
| 15.1 | `... latency --out runs/m15/l151-main --ns 1,4,16,32 --budgets 512,2048,4096`, then `... report --out runs/m15/l151-main --budget 8192 --latency 20` | 0.2 GPU-h | measured latency per question per arm; the recommendation |
| 15.2 | `python -m frontierlab.ttc.hf_spec own --out runs/m15/l152-main --gammas 1,2,4,6 --prompts 64 --max-new 128` | 0.3–0.6 GPU-h | acceptance of Qwen3-0.6B-Base drafting for 1.7B-Base; measured speed-up of the course loop at batch 1 |
| 15.2 | `... vllm --out runs/m15/l152-main --method draft_model --k 4`; `--method eagle3 --k 3`; `--method ngram --k 4` | 0.2 GPU-h | engine acceptance counters; tokens/s at batch 1 and 32 with and without speculation; does `LLM.get_metrics()` expose the counters at 0.30.0? |
| 15.3 | `vllm bench throughput --model Qwen/Qwen3-1.7B-Base --revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 --input-len 8192 --output-len 256 --num-prompts 200` and `--input-len 512 --output-len 4096` | < 0.5 GPU-h | measured tokens/s and vLLM's reported KV capacity vs the roofline (calibrates the simulator's efficiency) |
| 15.4 | the two `vllm bench throughput` runs with `--kv-cache-dtype auto` and `fp8`, `--output-len 8192 --num-prompts 256` | < 0.5 GPU-h | throughput gain from FP8 KV at long outputs; quality via `frontierlab.pipeline.hf_eval score` on each |
| project | lesson 15.1's commands for seeds 0 and 1 (`--seed 1` into `runs/m15/project-main-s1`), then `report` at `--latency 10, 20, 40` | 4–7 GPU-h | is the recommendation stable across seeds? |

Scaled pilot per plan section 12.1 (row "15.1 Test-time compute"): Qwen3-0.6B and Qwen3-1.7B on an A100 with vLLM,
100 questions, n = 16, the ORM only (outcome verifiers), then the PRM if memory allows; the pilot question is whether
the budget-matched ranking holds at both sizes. Main-path GPU-hours = generated tokens / measured tokens/s, labelled
PROJECTED. Suggested `curriculum/pilots/RESULTS.md` row: "P15 — Module 15 test-time compute and speculative decoding
(~5–8 H100-h PROJECTED) | not piloted (budget) | | | |".

## 7. BUILD_PROGRESS / TODO_FOR_TAL

- Free-CPU timings in Module 15 were measured with Module 16's jobs running on the same CPU; the 15.1 latency model
  disagreed with end-to-end timings by up to 2× under that load (stated in the lesson).
- Lesson 15.1's free-CPU world deliberately includes a programmatic checker (exact for this toy task); the lesson
  labels it as available only because the task allows it.
- The T4 variant of 15.1 needs a pinned revision of `Qwen/Qwen3-0.6B` (thinking model) at the pilot.
- AngelSlim's EAGLE-3 head ships a custom licence; read it before using the head in a public course artefact.
