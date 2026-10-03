# Open-model architectures — snapshot of October 2026

Dated appendix (plan principle 10). Lessons teach mechanisms that do not depend on this table; it records
what released open models looked like when the course was written, so a new release can be compared with
them. Lesson 01.2 explains how to read every column.

Every number was checked on 2026-10-03 against the `config.json` snapshot at the commit listed (stored in
`labs/common/frontierlab/calc/snapshots/`) and, where a report states it, against the report section given.
"Params (Transformers)" is the exact count of `AutoModelForCausalLM.from_config` on the meta device with
transformers 5.18.0 (no MTP modules); "reported" is the lab's own number and convention. KV per token is
the course calculator's value at a 32K context, BF16, one sequence.

| Model (HF repo @ commit) | Attention kind | Layers | Q heads / KV heads | Head dim | Context (config) | Experts total / active (+shared) | Norm / logit controls | Params: Transformers / reported | KV per token @32K | Sources (config key; report section) |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-ai/DeepSeek-V3 @e815299 | MLA, latent 512 + RoPE key 64 | 61 (3 dense FFN, 58 MoE) + 1 MTP | 128 / — (MLA; `num_key_value_heads` 128 is nominal) | qk 192 (128 + 64), v 128 | 163,840 | 256 / 8 (+1); sigmoid scores, aux-loss-free | RMSNorm on q and kv latents | 671.03B / 671B total, 37B activated | 68.6 KiB | `kv_lora_rank`, `qk_rope_head_dim`, `first_k_dense_replace`, `n_routed_experts`, `num_nextn_predict_layers`; V3 report §2.1.1, §4.2 |
| moonshotai/Kimi-K2-Instruct @fd1984e | MLA (as DeepSeek-V3) | 61 (1 dense, 60 MoE) | 64 / — | qk 192, v 128 | 131,072 | 384 / 8 (+1) | MuonClip during training (not in config) | 1,026.41B / 1.04T total, 32.6B activated (Table 2; convention not stated) | 68.6 KiB | `n_routed_experts`, `num_attention_heads`, `first_k_dense_replace`; K2 report §2.1, §2.3 Table 2 |
| Qwen/Qwen3-8B @b968826 | GQA | 36 | 32 / 8 | 128 | 40,960 (card: 32,768 native, 131,072 with YaRN) | dense | QK-norm (per head), no QKV bias | 8.19B / 8.2B | 144 KiB | `num_key_value_heads`, `head_dim`; Qwen3 report §2, Table 1; HF card |
| Qwen/Qwen3-235B-A22B @8efa617 | GQA | 94 | 64 / 4 | 128 | 40,960 | 128 / 8 (no shared) | QK-norm (per head) | 235.09B / 235B, 22B activated | 188 KiB | `num_experts`, `num_experts_per_tok`; Qwen3 report §2, Table 2 |
| Qwen/Qwen3-Next-80B-A3B-Instruct @9c7f2fb | hybrid: 36 Gated DeltaNet + 12 gated attention (3:1) | 48 | attention 16 / 2; DeltaNet 16 QK / 32 V heads | attention 256 (rotary 64); DeltaNet 128 | 262,144 | 512 / 10 (+1, gated) | QK-norm, partial RoPE 0.25, output gate in q_proj | 79.67B / 80B, 3B activated | 25.2 KiB (state of linear layers counted) | `full_attention_interval`, `linear_num_value_heads`, `partial_rotary_factor`, `num_experts`; HF model card |
| openai/gpt-oss-120b @b5c939d | GQA; alternating 128-token sliding and full | 36 (18 + 18) | 64 / 8 | 64 | 131,072 (YaRN ×32 from 4,096) | 128 / 4 | learned attention sinks; clamped SwiGLU (`swiglu_limit` 7.0); QKV and output biases | 116.83B / 116.83B total, 5.13B active (Table 1: unembedding in, embedding out) | 36.1 KiB | `layer_types`, `sliding_window`, `num_local_experts`, `initial_context_length`; model card Table 1, §2.2 |
| unsloth/gemma-3-27b-pt @eb493e0 (mirror of gated google/gemma-3-27b-pt) | GQA; 5 local (1024) : 1 global | 62 (52 + 10) | 32 / 16 | 128 | 131,072 | dense (GeGLU) | QK-norm replaces soft-capping (both softcaps null); pre- and post-norms; tied embeddings | 27.01B text / Table 1: 1,416M embedding + 25,600M non-embedding + 417M vision | 93.0 KiB | `sliding_window_pattern`, `sliding_window`, `attn_logit_softcapping`; Gemma 3 report §2, Table 1 (embedding differs from config's 1,409.6M; unexplained) |
| allenai/OLMo-2-1124-7B @7df9a82 | MHA | 32 | 32 / 32 | 128 | 4,096 | dense | QK-norm over all heads; norm on sublayer outputs (reordered norm); z-loss 1e-5 in training | 7.30B / 7B | 512 KiB | `num_key_value_heads`; OLMo 2 report §2.1, Table 1, §3.3.2 |
| zai-org/GLM-4.5 @cbb2c7c | GQA, partial RoPE 0.5 | 92 (3 dense, 89 MoE) + 1 MTP | 96 / 8 | 128 | 131,072 | 160 / 8 (+1) | QK-norm (per head); QKV bias | 352.80B / 355B total, 32B activated (Table 1: MTP in, embeddings and output layer out) | 368 KiB | `first_k_dense_replace`, `n_routed_experts`, `use_qk_norm`, `partial_rotary_factor`; GLM-4.5 §2.1, Table 1 |
| MiniMaxAI/MiniMax-M2 @757303d | GQA, full attention in every layer | 62 | 48 / 8 | 128 (rotary 64) | 196,608 | 256 / 8 (no shared) | QK-norm over all heads (`qk_norm_type` per_layer); MTP (3 modules in config) | 228.69B / 230B total, 10B active (HF card; convention not stated) | 248 KiB | `attn_type_list` (all 1), `num_local_experts`; HF card; full-attention rationale in MiniMax blog 2025-10-30 (no technical report) |

## Notes


- "Active" conventions differ by lab (see lesson 01.2). Recompute from config before comparing.
- Kimi K2's 32.6B activated does not match either convention exactly from the config (31.69B with head and no embedding, 32.86B with both); the report does not state its convention. Listed as an open discrepancy.
- Gated repos (google/gemma-3-*, meta-llama/*) return HTTP 401 to `raw/main/config.json` without accepting the licence; the snapshot uses an unofficial mirror whose shapes match the Gemma 3 report.
- Regenerate the numeric columns with `python -m frontierlab.calc --all --seq 4096 --context 32768`.
