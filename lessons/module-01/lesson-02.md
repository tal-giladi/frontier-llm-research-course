---
id: "01.2"
module: 1
minutes: 35
practice_minutes: 90
prerequisites: ["01.1"]
objectives:
  - Say which section of a technical report can support which kind of claim (architecture, data, cost, ablation, evaluation), and spot a claim that goes beyond its section.
  - Reconstruct an open model's architecture from its Hugging Face config.json and name the config key behind every number.
  - Compute total and active parameters, training FLOPs per token and KV-cache bytes per token for GQA, MLA, local-global and hybrid linear-attention models, and match published totals exactly.
  - Label claims about open and closed models as PUBLICLY DOCUMENTED, REASONABLE INDUSTRY PRACTICE, INFERENCE/SPECULATION or (company claim), and tag techniques ESTABLISHED, PROMISING or MODEL-SPECIFIC.
volatility: implementation
sources:
  - title: "DeepSeek-V3 Technical Report (section 2.1.1 MLA; section 4.2 hyper-parameters: 61 layers, d_c 512, d_h^R 64, 1 shared + 256 routed experts, 8 active, 671B/37B)"
    url: https://arxiv.org/abs/2412.19437
  - title: "DeepSeek-V2 (section 2.1.4, Table 1: KV cache per token for MHA, GQA, MQA, MLA)"
    url: https://arxiv.org/abs/2405.04434
  - title: "gpt-oss-120b & gpt-oss-20b Model Card (Table 1 parameter breakdown; section 2.2 architecture)"
    url: https://arxiv.org/abs/2508.10925
  - title: "Qwen3 Technical Report (section 2, Tables 1-2)"
    url: https://arxiv.org/abs/2505.09388
  - title: "Gemma 3 Technical Report (section 2, Table 1; section 5.2 local:global ablations)"
    url: https://arxiv.org/abs/2503.19786
  - title: "Kimi K2: Open Agentic Intelligence (section 2.3, Table 2)"
    url: https://arxiv.org/abs/2507.20534
  - title: "GLM-4.5 (section 2.1, Table 1: parameter counts include MTP, exclude embeddings and output layer)"
    url: https://arxiv.org/abs/2508.06471
  - title: "GPT-5 System Card (section 1: fast model, thinking model and real-time router; no architecture disclosed)"
    url: https://cdn.openai.com/gpt-5-system-card.pdf
  - title: "Gemini 3 Pro Model Card (sparse mixture-of-experts, natively multimodal, 1M-token context)"
    url: https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Pro-Model-Card.pdf
  - title: "Gemini 2.5 technical report (section 2.1: sparse MoE transformers with native multimodal support)"
    url: https://arxiv.org/abs/2507.06261
  - title: "The Llama 3 Herd of Models (section structure; section 3.2.1 scaling laws)"
    url: https://arxiv.org/abs/2407.21783
last_verified: "2026-10-03"
---

# 01.2 · Reading reports and configs as evidence

A technical report is a mix of things you can check, things the authors measured and you cannot repeat, and things they chose not to say. This lesson shows how to tell them apart: which section of a report can support which claim, how to rebuild an architecture from the config.json that ships with the weights, how to turn that config into parameters, FLOPs and KV-cache bytes with a calculator you can verify to the last parameter, and how to label what you write so a reader knows how much to trust it.

## Why this matters at a frontier lab

Most architecture decisions start with "lab X did Y". Before anyone spends GPU-hours on Y, someone has to answer three questions. What exactly did X do (the config, not the blog post)? What does it cost (parameters, FLOPs per token, cache bytes at our context length)? And how strong is the evidence that Y caused the result X reports? A research engineer who answers these in an afternoon, with every number traced to a file or a section, saves the team from copying a choice that only made sense under someone else's constraints. One who gets the arithmetic wrong — counting total instead of active parameters, or forgetting that half the layers keep only a 128-token window — produces a cost estimate that is off by an order of magnitude.

## The idea

### What each part of a report can prove

Technical reports have a stable shape. The DeepSeek-V3 report is typical: 1 Introduction; 2 Architecture (2.1.1 MLA, 2.1.2 DeepSeekMoE with auxiliary-loss-free balancing, 2.2 multi-token prediction); 3 Infrastructures (cluster, DualPipe, FP8 training, inference, hardware suggestions); 4 Pre-training (data, hyper-parameters, long-context extension, evaluations, discussion); 5 Post-training; 6 Conclusion and limitations; appendices with ablations. The Llama 3 report has the same skeleton plus inference, vision and speech sections, and real scaling-law experiments in section 3.2.1.

Each part supports a different kind of statement:

| Part | What it can establish | What it cannot |
|---|---|---|
| Architecture section + released config | what the model *is*: shapes, layer pattern, expert counts | that the choice was *better* than an alternative |
| Hyper-parameter section | the training recipe (tokens, LR, batch, schedule) | that the recipe is optimal or transfers to your scale |
| Infrastructure section | how they ran it; reported throughput and GPU-hours | your cost (different hardware, software, utilisation) |
| Ablation sections and appendices | that A beat B *in their setting*, usually at a smaller scale than the final model | that the gap holds at the final scale, or on your data |
| Evaluation tables | scores under their prompts, decoding and harness | a fair comparison with a model evaluated elsewhere |
| Discussion / limitations | what the authors believe | anything, without the evidence above |

The most common reading mistake is to take a claim from one row as if it came from another: "DeepSeek uses MLA, so MLA is better" turns an architecture fact into an ablation claim nobody in that report made.

### The config is the most reliable page

A Hugging Face `config.json` is a file the inference code actually reads. If it disagrees with the prose, the model you download behaves like the config. It is also small, versioned (every repo has a commit hash) and the same shape for every model of a family. But it has three blind spots:

1. **Facts that live in the modelling code.** The DeepSeek-V3 config has no field saying "attention is MLA" — it has `kv_lora_rank`, and the class `DeepseekV3Attention` decides what that means. Gemma 3's config does not say it has four norms per layer; `Gemma3DecoderLayer` does.
2. **Defaults.** Gemma 3's `text_config` has no `tie_word_embeddings` field; the class default is `True`. Absent is not the same as false.
3. **Things that are not shipped.** DeepSeek-V3's `num_nextn_predict_layers: 1` says there is a multi-token-prediction module, but the Transformers class does not build it, so a parameter count of the loaded model leaves it out.

The calculator in `frontierlab.calc` records, for every field, the config key it came from or "model_type (…)" when the fact comes from the modelling code. That is the citation you put in a table.

### Four numbers from a config

Symbols for one layer: $C$ hidden width, $H$ query heads, $K$ key/value heads, $d$ head dimension (query/key), $d_v$ value dimension, $V$ vocabulary, $L$ layers. For a mixture-of-experts (MoE) layer: $E$ routed experts, $k$ active per token, $I_e$ expert width, $s$ shared experts.

**Attention parameters (GQA).** $q$: $C \cdot Hd$; $k$: $C \cdot Kd$; $v$: $C \cdot K d_v$; output: $H d_v \cdot C$; plus small extras (biases, QK-norm gains, sink logits).

**Attention parameters (MLA, DeepSeek-V2/V3).** Queries go through a low-rank bottleneck of width $r_q$; keys and values are compressed into a shared latent of width $d_c$ plus one extra key of width $d_R$ that carries RoPE and is shared by all heads (DeepSeek-V2 section 2.1.3):

$$P_{\text{MLA}} = \underbrace{C r_q + r_q + r_q H (d_{\text{nope}} + d_R)}_{\text{queries}} + \underbrace{C(d_c + d_R) + d_c + d_c H (d_{\text{nope}} + d_v)}_{\text{keys and values}} + \underbrace{H d_v C}_{\text{output}}$$

**FFN parameters.** SwiGLU (and GeGLU) has three matrices: $3 C I$. An MoE layer has a router ($E \cdot C$), $E$ experts of $3 C I_e$ each and $s$ shared experts; a token uses the router, the shared experts and only $k$ routed experts.

**Active parameters** are what one token touches. Labs disagree on whether the input embedding (a lookup, no multiply) and the output head count. gpt-oss's Table 1 counts the unembedding but not the embedding; the Qwen3-30B-A3B card's "3.3B activated" matches counting both and its "29.9B non-embedding" matches counting neither; GLM-4.5's Table 1 includes the multi-token-prediction layer but excludes embedding and output layer. Before comparing two "active" numbers, find out which convention each used.

**Training FLOPs per token** (parent course lesson 07.4: a multiply-add is 2 FLOPs, training ≈ 3 × forward):

$$F_{\text{train}} \approx 3\Big(2 P_{\text{matmul}} + \sum_{\text{layers}} 2H(d + d_v)\,\bar k\Big)$$

where $P_{\text{matmul}}$ is the active parameters used in matrix multiplies (head included, embedding lookup and norm gains excluded) and $\bar k$ is the average number of keys a query attends to: $T/2$ for a full causal layer of length $T$, about $w - w^2/(2T)$ for a sliding window $w < T$, and no such term for a linear-attention layer (its fixed-size state update is counted separately and is small).

**KV-cache elements per token**, per layer: $K(d + d_v)$ for a GQA layer (for $d_v = d$ that is the $2Kd$ of lesson 01.1); $d_c + d_R$ for MLA, because only the latent and the shared RoPE key are cached (DeepSeek-V2 Table 1); at most $w$ tokens' worth for a sliding layer; a constant-size state for a linear layer. Multiply by bytes per element (2 for BF16, 1 for FP8).

### What closed labs disclose

Closed models are evidence only for what their documents say:

- **GPT-5 System Card** (OpenAI, 2025-08-13), section 1: GPT-5 is "a unified system" with a fast model (`gpt-5-main`), a deeper reasoning model (`gpt-5-thinking`) and a real-time router trained on signals such as model switches, preference rates and measured correctness; mini versions take over when usage limits are reached. The card discloses no parameter count, architecture or training compute.
- **Gemini 3 Pro Model Card** (Google DeepMind, released November 2025): a sparse mixture-of-experts transformer with native multimodal support for text, vision and audio; up to a 1M-token context and 64K output tokens; trained on TPUs with JAX and ML Pathways. No parameter counts, expert counts, layer counts or training compute. The Gemini 2.5 report (section 2.1) uses almost the same sentence.

So "Gemini 3 Pro is a sparse MoE" is PUBLICLY DOCUMENTED (company claim); "GPT-5 is an MoE" is INFERENCE/SPECULATION; "GPT-5 has N parameters" is not something any OpenAI document supports.

## Worked example

### DeepSeek-V3 from its config, by hand

From `deepseek-ai/DeepSeek-V3` config.json (commit `e815299`; the same values are in report section 4.2): $C = 7168$, $L = 61$, $H = 128$, $r_q$ = `q_lora_rank` = 1536, $d_c$ = `kv_lora_rank` = 512, $d_{\text{nope}}$ = `qk_nope_head_dim` = 128, $d_R$ = `qk_rope_head_dim` = 64, $d_v$ = `v_head_dim` = 128, $V$ = 129,280, `first_k_dense_replace` = 3 dense layers with `intermediate_size` 18,432, then 58 MoE layers with `n_routed_experts` 256, `num_experts_per_tok` 8, `n_shared_experts` 1, `moe_intermediate_size` 2048; untied head.

One MLA block:

- queries: $7168 \cdot 1536 + 1536 + 1536 \cdot 128 \cdot 192 = 11{,}010{,}048 + 1{,}536 + 37{,}748{,}736$
- keys and values: $7168 \cdot 576 + 512 + 512 \cdot 128 \cdot 256 = 4{,}128{,}768 + 512 + 16{,}777{,}216$
- output: $128 \cdot 128 \cdot 7168 = 117{,}440{,}512$

Total $187{,}107{,}328$ per layer, $11.41$B over 61 layers. Note the output projection is the largest piece.

One expert: $3 \cdot 7168 \cdot 2048 = 44{,}040{,}192$. An MoE layer holds the router ($256 \cdot 7168 = 1{,}835{,}008$) and 257 experts: $11{,}320{,}164{,}352$. A dense layer: $3 \cdot 7168 \cdot 18432 = 396{,}361{,}728$ — exactly 9 experts' worth, so a token does the same FFN work in every layer.

Totals: embedding and head $2 \cdot 129280 \cdot 7168 = 1{,}853{,}358{,}080$; attention $11{,}413{,}547{,}008$; dense FFN $1{,}189{,}085{,}184$; MoE $58 \cdot 11{,}320{,}164{,}352 = 656{,}569{,}532{,}416$; norms $881{,}664$. Sum: **671,026,404,352**, the "671B" of section 4.2 — and exactly the count Transformers 5.18.0 gives for the same config.

Active per token: head $926{,}679{,}040$ + attention + dense FFN + $58 \cdot (9 \cdot 44{,}040{,}192 + 1{,}835{,}008)$ + norms = **36.63B**; with the input embedding, 37.55B. The report says "37B activated" and does not say which convention; both round or truncate to 37.

FLOPs per training token at $T = 4096$: $2 P_{\text{matmul}} = 73.25$ GFLOP; attention scores $61 \cdot 2 \cdot 128 \cdot (192 + 128) \cdot 2048 = 10.23$ GFLOP; forward 83.5, training **250.5 GFLOP per token**. The shortcut $6 \times$ active ($219.8$) misses 12% here, and the gap grows with $T$.

KV cache: $512 + 64 = 576$ elements per token per layer, $576 \cdot 61 = 35{,}136$ elements, **70,272 bytes per token in BF16** (68.6 KiB). If the same 128 heads cached full keys and values, it would be $128 \cdot (192 + 128) \cdot 61 = 2{,}498{,}560$ elements — 71 times more. The report gives the mechanism (section 2.1.1: only $c^{KV}$ and $k^R$ are cached); the per-token number is our derivation.

### A cost check you can do from a report

Section 1 gives 2,664K H800 GPU-hours for pre-training on 14.8T tokens. Our arithmetic gives $14.8 \times 10^{12} \cdot 250.5 \times 10^9 = 3.71 \times 10^{24}$ FLOPs (ignoring the MTP module). Assuming the H800's dense BF16 peak equals the H100 SXM's $989$ TFLOP/s (the H800 is an H100 variant with reduced interconnect bandwidth; check the datasheet), that implies a utilisation of $3.71 \times 10^{24} / (2.664 \times 10^{6} \cdot 3600 \cdot 989 \times 10^{12}) \approx 39\%$. Measured against the FP8 peak (twice as high; the model trained mostly in FP8) it is about 20%. This is INFERENCE: it depends on our FLOP convention, the sequence length we assumed and which peak is fair. It is useful as a sanity check (a utilisation of 300% would mean an arithmetic or reporting error), not as a fact about DeepSeek's cluster.

### Tiny numbers: the sliding-window average

A sliding window of $w = 4$ in a sequence of $T = 8$. Query $t$ ($t = 0..7$) sees $\min(t+1, 4)$ keys: $1, 2, 3, 4, 4, 4, 4, 4$, mean $30/8 = 3.75$. The continuous formula gives $w - w^2/(2T) = 4 - 16/16 = 3$; for the full layer it gives $T/2 = 4$ against an exact mean of $4.5$. The formula drops the "+1" of the diagonal, which matters only when $T$ is tiny. At gpt-oss's $w = 128$, $T = 4096$: $128 - 128^2/8192 = 126$ keys on average, against 2,048 for a full layer — a 16× cheaper score computation in the sliding layers.

## Shapes and cost

| Model (config) | Attention, layers | Total / active | Train GFLOP/token, T = 4096 | KV per token at 32K, BF16 |
|---|---|---|---|---|
| Qwen3-8B | GQA 32/8, $d$ 128; 36 full | 8.19B / 7.57B | 49.0 | 144 KiB |
| OLMo-2-1124-7B | MHA 32/32, $d$ 128; 32 full | 7.30B / 6.89B | 44.6 | 512 KiB |
| Gemma 3 27B | GQA 32/16, $d$ 128; 52 sliding (1024) + 10 full | 27.01B / 27.01B (tied) | 165.3 | 93 KiB |
| gpt-oss-120b | GQA 64/8, $d$ 64; 18 sliding (128) + 18 full | 116.83B / 5.13B | 32.7 | 36 KiB |
| Qwen3-Next-80B-A3B | 36 Gated DeltaNet + 12 gated GQA 16/2, $d$ 256 | 79.67B / 3.56B | 22.9 | 25 KiB |
| DeepSeek-V3 | MLA 128 heads, latent 512 + 64; 61 full | 671.03B / 36.63B | 250.5 | 68.6 KiB |

All numbers are from `python -m frontierlab.calc --all` on the bundled snapshots; totals equal Transformers 5.18.0's parameter count exactly for all twelve snapshots (measured 2026-10-03, CPU, meta device). Active counts the head and not the input embedding.

Three things the table shows. Active parameters, not total, set FLOPs per token: gpt-oss-120b has 14× more parameters than Qwen3-8B but costs less per token. KV bytes depend on attention design far more than on size: OLMo 2 7B caches 3.5× more per token than Qwen3-8B because it keeps all 32 key/value heads. And per-token KV for windowed and hybrid models depends on the context length you quote: Gemma 3 at 128K caches 10.4 GiB per sequence, against 62 GiB if all 62 layers were global.

Data types and devices: the calculator is pure Python integers on the CPU; nothing is allocated. `measure_cache.py` in the lab builds shrunk models (2–6 layers, random fp32 weights, CPU) whose cache tensors have shape (B, K, tokens, $d$), fp32 on CPU, so you can see the formula hold in real code.

## Build it

The package is `frontierlab.calc`: `hfconfig.py` reads a config into an `ArchSpec` (one vocabulary for every family, with `spec.source` naming the config key behind each field) and `arith.py` does the arithmetic.

```python
from frontierlab.calc import from_hf, param_counts, flops_per_token, kv_bytes_per_token

s = from_hf("deepseek-ai/DeepSeek-V3")        # bundled snapshot; from_hf(fetch_config(repo)) for the live file
print(s.source["n_experts"])                  # 'n_routed_experts'  -> what you cite
pc = param_counts(s)
print(pc["total"], pc["active"])              # 671026404352 36625603584
print(flops_per_token(s, T=4096) / 1e9)       # 250.45
print(kv_bytes_per_token(s, S=32768))         # 70272.0
```

```bash
python -m frontierlab.calc --all                                   # every snapshot
python -m frontierlab.calc openai/gpt-oss-20b --fetch --context 131072
```

The correctness check is external, not self-referential: `labs/common/tests/test_calc.py` builds every snapshot with `AutoModelForCausalLM.from_config` on PyTorch's meta device (no memory is allocated) and requires the calculator's total to equal Transformers' count to the parameter. It also checks the published active counts (gpt-oss 5.13B and 3.61B; Qwen3-30B-A3B 3.3B), the Baseline-0 numbers from lesson 01.1, and that sliding layers stop growing at the window.

An exact match is a strong test because the config does not contain everything. Leave out Gemma 3's two extra norms per layer (the modelling code has four, the config says nothing) and the total is short by $2 \cdot 62 \cdot 5376 = 666{,}624$; treat OLMo 2's QK-norm as per-head like Qwen3's instead of one gain over all heads ($Hd$ and $Kd$ wide) and it is short by $32 \cdot (2 \cdot 4096 - 2 \cdot 128) = 253{,}952$. Both errors are invisible at "7.3B" precision and both show up immediately against the reference. The lab's `measure_cache.py` found one more gap: Transformers 5.18.0 keeps $w - 1$ tokens in a sliding layer between steps (127 for gpt-oss, 1023 for Gemma 3) because the incoming token is the $w$-th. The calculator counts $w$, the number of keys each query attends to.

## What the evidence says

- **ESTABLISHED:** GQA (Llama 3, Qwen3, Gemma 3, gpt-oss, GLM-4.5, MiniMax-M2 all use it); QK-norm (Qwen3 section 2, Gemma 3 section 2, OLMo 2 section 2.1, GLM-4.5 section 2.1); fine-grained MoE with a small number of active experts (DeepSeek-V3, Qwen3, Kimi K2, GLM-4.5, gpt-oss).
- **PROMISING:** MLA — used by DeepSeek and adopted by Kimi K2 (section 2.3, Table 2), with evidence mostly from DeepSeek's own reports; local:global interleaving — Gemma 3 section 5.2 reports "minimal impact on perplexity" across ratios, measured on 2B text-only models, and gpt-oss uses a different version (1:1, 128-token band).
- **MODEL-SPECIFIC:** the exact hybrid ratio of Qwen3-Next (3 Gated DeltaNet : 1 gated attention, HF model card); learned attention sinks (gpt-oss section 2.2); 64 instead of 128 heads (Kimi K2 section 2.3, motivated by an 83% inference-FLOPs saving at 128K context).
- **Unresolved discrepancy:** Gemma 3 Table 1 lists 1,416M embedding parameters for the 27B model; the config gives $262{,}208 \cdot 5{,}376 = 1{,}409.6$M. The report does not explain the 6.4M gap. Cite the table for the reported number and the config for the computed one.
- **Closed models:** see "What closed labs disclose" above. Anything more is INFERENCE.
- A dated comparison of ten open models' architecture fields, each traced to a config key and a report section, is kept with the course's frontier-model reference material and is refreshed as models are released.

## Lab

> [!NOTE]
> This lab reads documents and does arithmetic; it compares nothing, so it has no experiment contract. Lesson 01.3 writes the first one.

**Folder:** [`labs/module-01/lesson-02/`](../../labs/module-01/) · **Time:** about 90 minutes · **Pass check:** `pytest labs/module-01/lesson-02` passes, `measure_cache.py` agrees with your formula (up to the $w-1$ effect), and your `claims.md` labels match the reference for at least 8 of 10 claims.

| Variant | Hardware | What you run |
|---|---|---|
| Main path | any machine; no GPU needed (this lab is arithmetic). Optional on a rented GPU: step 6 | steps 1–5; step 6 if you have a GPU session open for another lab |
| Free GPU (Colab/Kaggle T4) | T4 | steps 1–5 (step 6 at a shorter context fits in 16 GB) |
| Free CPU | laptop; tests measured at 11 s, `measure_cache.py` at 80 s (mostly importing Transformers), 2026-10-03 | steps 1–5 |

1. **Rebuild one model by hand.** Pick Qwen3-30B-A3B or gpt-oss-20b. Open its config with `python labs/module-01/lesson-02/reconstruct.py Qwen/Qwen3-30B-A3B --fetch` (or without `--fetch` to use the dated snapshot), and on paper compute total parameters, active parameters and KV bytes per token at 32K. Then compare with the script's output.
2. **Implement the three functions** in `lab.py`: `attention_params` (GQA and MLA), `kv_elements` (full, sliding and MLA layers) and `active_params`. The tests compare your numbers with Transformers' exact counts for twelve models.
3. **Check the formula against real code.** Run `python labs/module-01/lesson-02/measure_cache.py`. Explain in two sentences why the sliding layers store 127 and 1023 tokens, not 128 and 1024, and whether that changes any decision.
4. **Read one report against its config.** Open the gpt-oss model card (arXiv 2508.10925) Table 1 and section 2.2 next to `openai/gpt-oss-120b` config.json. Find the config key for each architecture claim in section 2.2, and the one claim in the config that the card text does not state (hint: look at `initial_context_length`).
5. **Label the evidence.** Fill in `labs/module-01/lesson-02/claims.md`.
6. **Optional, main path.** `python labs/module-01/lesson-02/gpu_cache_check.py --model Qwen/Qwen3-0.6B --context 32768` loads the real model in BF16, runs a 32K-token prefill with `use_cache=True`, and prints the cache bytes and the allocator growth next to the calculator's prediction. Not run in this build; part of the Module 1 pilot.

<details>
<summary>Hint for step 2 (MLA)</summary>

Every MLA block has the same parameter count, and the RoPE key is one $d_R$-wide vector shared by all heads, so it appears once in `kv_a` (width $d_c + d_R$) and not in `kv_b`. The test's hand value for DeepSeek-V3 is 187,107,328.

</details>

<details>
<summary>Reference labels for claims.md</summary>

1. PUBLICLY DOCUMENTED mechanism (DeepSeek-V3 section 2.1.1) plus our arithmetic from config (`kv_lora_rank` 512 + `qk_rope_head_dim` 64). Config alone gives the number only if you know MLA caches the latent; strengthen by measuring an MLA cache in code.
2. PUBLICLY DOCUMENTED (gpt-oss model card Table 1) and reproducible from config to 0.01B. Note the convention: unembedding in, embedding out.
3. INFERENCE/SPECULATION. The GPT-5 system card discloses no architecture. Nothing short of an OpenAI disclosure strengthens it.
4. PUBLICLY DOCUMENTED (company claim), Gemini 3 Pro model card, Model Information. No config; no size or expert count is disclosed.
5. PUBLICLY DOCUMENTED (company claim), GPT-5 system card section 1. It describes the ChatGPT system, not one model.
6. PUBLICLY DOCUMENTED (Qwen3 report section 2: excludes shared experts) and consistent with the config (no shared-expert field; Transformers builds none).
7. Overstated. Gemma 3 section 5.2 reports minimal perplexity impact on 2B text-only models; that is the lab's own ablation (company claim) on one metric at one scale, not "no quality cost". Strengthen with long-context and downstream evaluations at larger scale.
8. PUBLICLY DOCUMENTED (company claim), Kimi K2 abstract and section 2. Training curves cannot be checked from outside.
9. Not a claim any cited report makes in this general form. MLA is PROMISING; whether it beats GQA at matched budget and context is exactly the kind of question Module 3 tests.
10. PUBLICLY DOCUMENTED with a convention: GLM-4.5 Table 1 includes the MTP layer and excludes embeddings and output layer. Transformers builds 352.8B without MTP and with embeddings; both are consistent once the convention is known.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-01/lesson-02/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-01/lesson-02`.

</details>

## Common mistakes

- **Comparing "active" numbers across labs without the convention.** Embedding in or out, head in or out, MTP in or out: the same model can be "3.0B", "3.3B" or "3.6B active".
- **Treating absent fields as false.** A missing `tie_word_embeddings` in Gemma 3 means "class default", which is true.
- **Using `max_position_embeddings` as "the context length".** Qwen3-8B's config says 40,960; the model card says 32,768 natively and 131,072 with YaRN. Cite the card for capability and the config for what the code allows.
- **Quoting one KV-per-token number for a windowed model.** It depends on the context; always say "at S tokens".
- **Citing an unofficial mirror as the source.** The Gemma 3 snapshot here is `unsloth/gemma-3-27b-pt` because Google's repo is gated; its shapes match the report, but a citation should name the official repo and report section.
- **Reading an ablation's scale as the model's scale.** Most ablations in reports are run on models much smaller than the one released.

## References

- DeepSeek-AI, *DeepSeek-V3 Technical Report*, sections 2.1, 4.2. https://arxiv.org/abs/2412.19437
- DeepSeek-AI, *DeepSeek-V2*, sections 2.1.2–2.1.4, Table 1. https://arxiv.org/abs/2405.04434
- OpenAI, *gpt-oss-120b & gpt-oss-20b Model Card*, Table 1, section 2.2. https://arxiv.org/abs/2508.10925
- Qwen Team, *Qwen3 Technical Report*, section 2, Tables 1–2. https://arxiv.org/abs/2505.09388
- Gemma Team, *Gemma 3 Technical Report*, section 2, Table 1, section 5.2. https://arxiv.org/abs/2503.19786
- Moonshot AI, *Kimi K2*, section 2.3, Table 2. https://arxiv.org/abs/2507.20534
- Zhipu AI, *GLM-4.5*, section 2.1, Table 1. https://arxiv.org/abs/2508.06471
- Meta, *The Llama 3 Herd of Models*, section 3.2.1. https://arxiv.org/abs/2407.21783
- OpenAI, *GPT-5 System Card*, section 1. https://cdn.openai.com/gpt-5-system-card.pdf
- Google DeepMind, *Gemini 3 Pro Model Card*. https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Pro-Model-Card.pdf
- Google DeepMind, *Gemini 2.5*, section 2.1. https://arxiv.org/abs/2507.06261
- Config snapshots and commit hashes: `labs/common/frontierlab/calc/snapshots/index.json` (fetched 2026-10-03). Software versions: [references/versions.md](../../references/versions.md).

## Next

[01.3 · Designing an experiment](lesson-03.md)
