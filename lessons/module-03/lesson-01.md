---
id: "03.1"
module: 3
minutes: 40
practice_minutes: 90
prerequisites: ["01.1", "01.2", "01.3", "01.4", "02.4"]
objectives:
  - Derive MLA's cache size per token (latent plus decoupled RoPE key) and compare it with MHA, GQA and MQA for a given configuration, by hand and with code.
  - Explain, with a numerical example, why RoPE on the up-projected keys prevents absorbing the key up-projection into the query, and how the decoupled RoPE key fixes it.
  - Implement weight-absorbed MLA attention and show it equals the naive path to float64 precision, in full forward and in cached decoding.
  - Run the full correctness suite (gradient, causal, cached decode, naive vs absorbed) before measuring anything, then measure decode memory and latency with the Module 2 method and state what the measurement does and does not show.
volatility: concept
sources:
  - title: "DeepSeek-V2 (section 2.1: MLA, low-rank key-value joint compression, decoupled RoPE; Table 1: KV cache per token; abstract: KV cache reduced by 93.3%)"
    url: https://arxiv.org/abs/2405.04434
  - title: "DeepSeek-V3 Technical Report (section 2.1.1: MLA as used in V3; config.json fields)"
    url: https://arxiv.org/abs/2412.19437
  - title: "Kimi K2: Open Agentic Intelligence (section 2.1: QK-Clip treatment of MLA's shared rotary key)"
    url: https://arxiv.org/abs/2507.20534
  - title: "Hugging Face Transformers, modeling_deepseek_v3.py (main branch, checked 2026-10-03: the cache stores the compressed latent)"
    url: https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py
  - title: "Shazeer — Fast Transformer Decoding: One Write-Head is All You Need (multi-query attention)"
    url: https://arxiv.org/abs/1911.02150
  - title: "Ainslie et al. — GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
    url: https://arxiv.org/abs/2305.13245
last_verified: "2026-10-03"
---

# 03.1 · Multi-head Latent Attention

Grouped-query attention saves decode memory by letting several query heads share one key and value head. Multi-head Latent Attention (MLA), introduced in DeepSeek-V2 and used in DeepSeek-V3 and Kimi K2, saves it differently: every token is compressed into one small latent vector, and each head's key and value are up-projections of that latent. This lesson derives MLA from its equations, shows why rotary position embeddings force a separate "decoupled" RoPE key, implements the weight-absorbed form that never builds per-head keys at all, checks that every form computes the same function, and then measures what MLA does to decode memory and latency.

## Why this matters at a frontier lab

At long context the KV cache, not the weights, decides how many sequences fit on a GPU and how fast each decode step runs: a decode step reads every cached byte once per generated token. Baseline-0 caches 12 KiB per token, 384 MiB for one 32K sequence. Large models are worse: a 128-head model with full multi-head attention would cache tens of MiB per thousand tokens per layer. DeepSeek-V2 reports that MLA cut its KV cache by 93.3% relative to DeepSeek 67B and raised maximum generation throughput 5.76× (abstract; company claim, against a different, dense model). That is why MLA is now a standard option, and why a research engineer must be able to answer: for *our* model and serving constraint, does MLA beat a cheaper GQA, and how do we know the implementation is right? Getting it wrong is easy: an implementation that caches the up-projected keys "for speed" keeps all of GQA's memory cost and none of MLA's saving, and a naive decode path that re-expands the whole cache at every step can be slower than the model it replaced.

## The idea

### From GQA to a shared latent

Symbols: $h_t \in \mathbb{R}^C$ the attention input of token $t$ (after the block's RMSNorm), $H$ query heads, $d$ head width, $L$ layers.

- **MHA** caches $k_{t,i}, v_{t,i} \in \mathbb{R}^d$ for every head $i$: $2Hd$ elements per token and layer.
- **GQA** with $K$ key/value heads caches $2Kd$; **MQA** is $K = 1$, $2d$.
- **MLA** caches one latent $c_t \in \mathbb{R}^{d_c}$ per token and builds every head's key and value from it (DeepSeek-V2 section 2.1.2):

$$c_t = W^{DKV} h_t, \qquad k^{C}_{t,i} = W^{UK}_i c_t, \qquad v_{t,i} = W^{UV}_i c_t$$

$W^{DKV} \in \mathbb{R}^{d_c \times C}$ is the down-projection, $W^{UK}_i \in \mathbb{R}^{d_n \times d_c}$ and $W^{UV}_i \in \mathbb{R}^{d_v \times d_c}$ the per-head up-projections ($d_n$ the "no-position" key width, $d_v$ the value width). Every head still has its own key and value, so MLA keeps multi-head expressiveness; what is shared is the *storage*. DeepSeek-V2 applies an RMSNorm to the latent (the report adds "additional RMS Norm layers after the compressed latent vectors"); so does our code and Hugging Face's (`kv_a_layernorm`). Queries can be compressed the same way (width $d_c'$); that saves activation memory in training, not cache, and is optional here (`q_lora_rank`).

### Weight absorption

The score of head $i$ between query $t$ and cached token $s$ is a dot product, so the up-projection can move to the query side:

$$q^{C\top}_{t,i}\, k^C_{s,i} = q^{C\top}_{t,i} W^{UK}_i c_s = \big(W^{UK\top}_i q^C_{t,i}\big)^{\!\top} c_s$$

Compute $\tilde q_{t,i} = W^{UK\top}_i q^C_{t,i} \in \mathbb{R}^{d_c}$ once per new token and head, and attend directly over the cached latents. Likewise the output of head $i$ is $\sum_s p_{ts} W^{UV}_i c_s = W^{UV}_i \sum_s p_{ts} c_s$: attend over the latents, then apply $W^{UV}_i$ once ("$W^{UK}$ can be absorbed into $W^{Q}$, and $W^{UV}$ can be absorbed into $W^{O}$", DeepSeek-V2 section 2.1.2). Over the latent, all heads use the same key and the same value — **absorbed MLA is multi-query attention with head width $d_c$**, with $H$ different queries. That is the form a decoding kernel uses.

### Why RoPE breaks absorption, and the decoupled key

RoPE rotates each key by a matrix $R_s$ that depends on its position. If it were applied to the up-projected key, $k^C_{s,i} = R_s W^{UK}_i c_s$, the score would be $q^\top R_s W^{UK}_i c_s$: the matrix between the query and the cached latent is now different for every cached position $s$, so there is no single matrix to fold into the query. DeepSeek-V2 section 2.1.3 states it directly: a RoPE matrix for the current token would sit between $W^{Q}$ and $W^{UK}$, "and matrix multiplication does not obey a commutative law". You would have to rebuild every cached key at every step, which is what the cache was meant to avoid.

The fix (section 2.1.3) adds a small, separate key that carries position, computed from $h_t$ and **shared by all heads**, and a matching per-head query part:

$$k^R_t = \mathrm{RoPE}(W^{KR} h_t) \in \mathbb{R}^{d_r}, \qquad q_{t,i} = [\,q^C_{t,i}\,;\, q^R_{t,i}\,], \qquad k_{t,i} = [\,k^C_{t,i}\,;\, k^R_t\,]$$

$$\text{score}_{ts,i} = \frac{q^{C\top}_{t,i} k^C_{s,i} + q^{R\top}_{t,i} k^R_s}{\sqrt{d_n + d_r}}$$

The content part is absorbed as above; the RoPE part is small and cached as it is. The cache per token and layer is $d_c + d_r$ elements ("the decoupled key should also be cached", section 2.1.3). DeepSeek-V2 sets $d_c = 4 d_h$ and $d_r = d_h/2$, giving $\tfrac{9}{2} d_h$ elements per layer, which Table 1 notes equals GQA with only 2.25 groups.

## Worked example

### Absorption with tiny numbers

One head, $d_n = 2$, $d_c = 3$:

$$W^{UK} = \begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & 1 \end{pmatrix}, \quad c_s = (1, 2, 0), \quad q = (3, 1)$$

Naive: $k = W^{UK} c_s = (1, 2)$, score $q \cdot k = 3 + 2 = 5$. Absorbed: $\tilde q = W^{UK\top} q = (3, 1, 3 \cdot 2 + 1) = (3, 1, 7)$, score $\tilde q \cdot c_s = 3 + 2 + 0 = 5$. Same number, and $\tilde q$ is computed once for all cached $s$.

### What RoPE does to it

Now rotate the key by the position-$s$ RoPE matrix; take a 90° rotation, $R_s = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$. Then $R_s k = (-2, 1)$ and the score is $3 \cdot (-2) + 1 = -5$. The fixed $\tilde q = (3, 1, 7)$ still gives 5. To get $-5$ we would need $\tilde q_s = (R_s W^{UK})^\top q = (1, -3, -1)$, and $\tilde q_s \cdot c_s = 1 - 6 + 0 = -5$ — a different $\tilde q$ for every cached position. `why_decoupled.py` in the lab repeats this with random matrices and real RoPE: max error $4 \times 10^{-15}$ without RoPE on $W^{UK} c$, about 23 (scores of size about 5) with it, and $4 \times 10^{-15}$ again with the decoupled key.

### Cache size by hand

Baseline-0 has $L = 12$, $H = 12$, $d = 64$, $K = 4$. Per token, in BF16 (2 bytes):

| Design | Elements per layer | Bytes per token (12 layers) | At 32K tokens |
|---|---|---|---|
| MHA ($K = 12$) | $2 \cdot 12 \cdot 64 = 1536$ | 36,864 | 1,152 MiB |
| GQA $K = 4$ (Baseline-0) | $2 \cdot 4 \cdot 64 = 512$ | 12,288 | 384 MiB |
| GQA $K = 2$ | 256 | 6,144 | 192 MiB |
| MLA $d_c = 256$, $d_r = 32$ | $256 + 32 = 288$ | 6,912 | 216 MiB |
| MQA ($K = 1$) | 128 | 3,072 | 96 MiB |

So at Baseline-0's size, DeepSeek's recipe ($d_c = 4d$, $d_r = d/2$) caches 56% of what Baseline-0 caches — but slightly *more* than GQA with two KV heads. Against MHA, the comparison DeepSeek-V2 makes, it is 19%. Against an already-grouped baseline the saving is modest, which is why the module project compares MLA with a KV-matched GQA and not only with Baseline-0. For DeepSeek-V2 itself ($H = 128$, $d_h = 128$, $d_c = 512$, $d_r = 64$, 60 layers): MLA caches 576 elements per layer against $2 \cdot 128 \cdot 128 = 32{,}768$ for MHA, 1.8%.

## Shapes and cost

Shapes for one MLA layer at Baseline-0 width ($C = 768$, $H = 12$, $d_n = d_v = 64$, $d_r = 32$, $d_c = 256$), batch $B$, $T$ new tokens, $S$ tokens in the cache. Main path: GPU, BF16 under autocast; free CPU: fp32. Names follow Hugging Face `DeepseekV3Attention`.

| Tensor | Shape | Notes |
|---|---|---|
| `q_proj` output | (B, T, 12 · 96) | split into $q^C$ (B, 12, T, 64) and $q^R$ (B, 12, T, 32); RoPE on $q^R$ |
| `kv_a_proj_with_mqa` output | (B, T, 256 + 32) | split into $c$ (B, T, 256), normed, and $k^R$ (B, 1, T, 32), RoPE |
| cache `c_kv`, `k_rope` | (B, S, 256), (B, 1, S, 32) | the only per-token state; plus `pos` (S,) int64 in our code |
| naive: `kv_b_proj(c)` | (B, 12, S, 64 + 64) | per-head $k^C$, $v$ rebuilt from the whole cache at every step |
| absorbed: $\tilde q$ | (B, 12, T, 256) | `einsum("bhtn,hnc->bhtc", q_nope, W_UK)` |
| absorbed: attention | queries (B, 1, 12·T, 288), key (B, 1, S, 288), value (B, 1, S, 256) | MQA over the latent, heads folded into query rows |

**Parameters per layer** (no query compression): $C \cdot H(d_n + d_r) + C(d_c + d_r) + d_c + d_c H (d_n + d_v) + H d_v C = 884{,}736 + 221{,}184 + 256 + 393{,}216 + 589{,}824 = 2{,}089{,}216$, against Baseline-0's 1,572,992 (test `test_mla_param_count_by_hand_baseline0_shape`). Over 12 layers MLA adds 6.2M parameters (+6.4% non-embedding), so "MLA vs Baseline-0" at equal tokens is also a comparison between models of different size; the project removes that with an equal-parameters arm (SwiGLU width 2,592 instead of 2,816) and an equal-FLOPs arm.

**Training FLOPs per token** at $T = 1024$ (`frontierlab.attention.accounting.flops_per_token`, same convention as lesson 01.1): Baseline-0 788.1M, MLA 839.5M. The training path is the naive one (each token's latent is expanded once, and that cost is in the parameter term).

**Decode FLOPs per token at $S = 32{,}768$**, per (query, key) pair and layer:

- GQA: $4Hd = 3{,}072$;
- naive MLA: $2H(d_n + d_r) + 2Hd_v = 3{,}840$, **plus** re-expanding every cached latent: $2 d_c H (d_n + d_v) = 786{,}432$ per cached token per layer — 311 GFLOP per generated token at 32K, 214× Baseline-0's 1.45 GFLOP;
- absorbed MLA: $2H(d_c + d_r) + 2Hd_c = 13{,}056$, no re-expansion: 5.39 GFLOP per token.

**Bytes read per decode step** decide time at batch 1, because decode attention is memory-bound: absorbed MLA does $13{,}056$ FLOPs per 576 bytes of cache read, an intensity of 22.7 FLOP/byte, far below the H100's ridge point of about 295 (lesson 02.1). A roofline lower bound for one decode step at 32K, BF16, H100 SXM (3.35 TB/s): weights plus cache, Baseline-0 $646.5$ MB → 193 µs; MLA $482.7$ MB → 144 µs; GQA $K = 2$ $440.4$ MB → 131 µs. PROJECTED from $t \ge \text{bytes}/\text{bandwidth}$, pending the Module 3 pilot; real kernels add launch and softmax costs, and only a fused MLA kernel gets near this bound.

## Build it

`labs/common/frontierlab/attention/mla.py` registers `"mla"`. Settings come from `cfg.extra` (`kv_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`, `v_head_dim`, `q_lora_rank`, `mla_mode`); `num_key_value_heads` is ignored and Baseline-0's QK-norm is not used — MLA normalises the latent instead, as DeepSeek does. That is a second difference from Baseline-0, and the project's contract has to name it.

The forward pass computes queries, the normalised latent and the RoPE key, appends the latter two to the `LayerCache` (`c_kv`, `k_rope`, `pos`), then runs one of two paths on the same cache:

```python
fn = self.absorbed if self.mode == "absorbed" else self.naive
y = fn(q_nope, q_rope, c, k_rope, positions, k_pos)            # (B, H, T, d_v)
```

`naive` rebuilds per-head keys and values with `kv_b_proj` and calls ordinary attention. `absorbed` folds $W^{UK}$ into the query, attends with one shared key/value over the latent — the $H$ heads are reshaped into $H \cdot T$ query rows, so the cache is read once rather than copied per head — and applies $W^{UV}$ to the result. `set_mla_mode(model, "absorbed")` switches every layer. Hugging Face's current implementation (main branch, checked 2026-10-03) caches the compressed latent and expands it at each step, like our naive path; serving engines and DeepSeek's FlashMLA library implement the absorbed form in fused kernels (their supported hardware changes between releases; check the README).

The correctness suite (`frontierlab.attention.checks.attention_suite`) must pass before any MLA number counts:

```text
PASS  op-level float64 gradcheck (input and 5 parameter tensors)
PASS  causal check: max |diff| = 0.00e+00
PASS  cached decode, 1 token(s) per step: max |diff| = 3.50e-15
PASS  cached decode, 5 token(s) per step: max |diff| = 2.76e-15
PASS  MLA naive vs absorbed (float64): max |diff| = 2.35e-15
PASS  cached decode in absorbed mode: max |diff| = 2.98e-15
```

Two details make these checks real. First, they run on a *sharpened* copy of the model (`checks.sharpen`: weights re-drawn with std 0.2): at the course's initialisation (std 0.02) attention is almost uniform and each layer's output is tiny next to the residual stream, so a 1% error in the absorbed path changed the logits by less than float64 rounding in our first attempt. Second, `frontierlab.layers.RMSNorm` normalises in float32 even for float64 inputs, which rounds every model-level float64 comparison to float32 resolution at each norm; the suite runs with RMSNorm in the input dtype (`checks.exact_rmsnorm`). With both in place, a planted 1% error in the absorbed path's scale shows up as a logit difference of $8.7 \times 10^{-3}$.

## What the evidence says

- **ESTABLISHED.** MLA is used by DeepSeek-V2, V3 and V3.x and by Kimi K2 (their configs: `kv_lora_rank` 512, `qk_rope_head_dim` 64), and by Kimi Linear's full-attention layers; it is PUBLICLY DOCUMENTED in DeepSeek-V2 section 2.1 with equations, and supported in Hugging Face Transformers, vLLM and SGLang.
- **The memory saving is arithmetic**, PUBLICLY DOCUMENTED (Table 1) and exact in our code (lab part 3). Whether it is a saving *for your model* depends on the baseline: against MHA it is large, against a GQA model with few KV heads it can be small or negative (the table above).
- **Quality.** DeepSeek-V2 reports that MLA performs better than MHA while caching much less (section 2.1 and Table 1 caption: "its performance is stronger than MHA"), with ablations in its Appendix D that this course has not re-checked. That is one lab's evidence at its scales (company claim). Whether MLA matches a KV-matched GQA at 100M parameters is a hypothesis the module project tests; expect the difference to be within the noise floor at course scale.
- **Interaction with other choices.** Kimi K2's QK-Clip rescales MLA's per-head query and key parts but leaves the shared rotary key $k^R$ untouched "to avoid effect across heads" (K2 section 2.1) — a reminder that shared components change how per-head interventions work.
- **Open questions.** How small $d_c$ can go before quality drops at a given scale; how MLA interacts with long-context RoPE scaling (only $d_r$ channels carry position); FP8 caches (DeepSeek-V4 stores the RoPE dimensions in BF16 and the rest in FP8, per its report) — Module 8 returns to low-precision caches.

## Lab

**Folder:** [`labs/module-03/lesson-01/`](../../labs/module-03/) · **Time:** about 90 minutes · **Pass check:** `pytest labs/module-03/lesson-01` passes; `check_mla.py` prints only PASS lines; your notes state the decision the contract's rule gives, with its numbers.

### Experiment contract

- **Question:** at contexts from 1K to 16K tokens (free CPU) or 8K to 32K (main path), how much decode-cache memory do MLA ($d_c = K d$, $d_r = d/2$) and KV-halved GQA save against Baseline-0's GQA, and what do MLA's two decode paths cost per generated token? Decision informed: which MLA decode path the module project uses, and whether the cache formula can be trusted for main-path projections.
- **Hypotheses and status:** (1) measured cache bytes equal the formula exactly — established (deterministic); (2) absorbed decode is faster than naive at long context, because naive re-expands every cached latent — established mechanism, size hardware-dependent; (3) MLA decode faster than GQA — may not appear: absorbed MLA does about 4× more FLOPs per key, and only memory-bound hardware turns its smaller cache into time.
- **Baseline:** arm `b0` (the preset's GQA), random weights, seed 0. Decode memory and time do not depend on the weight values.
- **Changed variable:** the attention kind (`gqa-kv`, `mla-naive`, `mla-absorbed`). **Controlled:** width, depth and query heads of the preset, vocabulary 8,192, batch 1, dtype, thread count, the same prefilled token ids, prefill chunk, warm-up and rounds.
- **Comparison axis:** equal model shape at equal context. Not equal parameters (toy MLA has 9% more non-embedding parameters), and it says nothing about quality or training cost — the project measures those.
- **Budget:** free CPU about 2 minutes; main path about 10 GPU-minutes on one H100 or A100.
- **Metrics and decision rule:** cache bytes (`Cache.nbytes`, exact) against `accounting.cache_bytes`; median decode-step time over 30 interleaved rounds after 3 warm-up rounds, with a 95% bootstrap interval, and the paired speed-up against `b0`. Rules stated now: the formula is trusted if measured and predicted bytes are identical for every arm and context. The project uses the absorbed path if, at the longest context, its speed-up interval against `b0` lies entirely above naive's; otherwise the result is "not distinguished on this hardware", and the project uses the absorbed path on the basis of its FLOP count, labelled as such.
- **Correctness checks:** `check_mla.py` (op-level float64 gradient check, causal check, cached decode with 1 and 5 tokens per step, naive vs absorbed in full forward and cached decode) passes before any timing.
- **Fallback evidence:** the Module 3 pilot's GPU traces, labelled as provided, if you have no GPU.
- **Limits:** random weights; one batch size; CPU results say nothing about GPU kernels; our implementation stores an int64 position per cached token per layer (reported separately); prefill cost is printed but not compared.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 SXM or A100 80 GB, about 10 GPU-minutes. Not run in this build; part of the Module 3 pilot | `python labs/module-03/decode_compare.py --device cuda --dtype bf16 --preset baseline0 --vocab 32768 --arms b0 gqa-kv mla-naive mla-absorbed --contexts 8192 16384 32768 --rounds 30 --out runs/l31/decode.json` |
| Free GPU (Colab/Kaggle T4) | T4, fp32 (no BF16 tensor cores) | the same with `--dtype fp32 --preset pilot-10m --contexts 4096 8192 16384`; you will not see Baseline-0's widths |
| Free CPU | laptop; `check_mla.py` about 20 s, `decode_compare.py` about 1 minute (measured 2026-10-03, 16 threads) | the steps below as written |

### Steps

1. **Implement** `expand_latent`, `absorb_query`, `absorbed_attention` and `kv_elements_per_token` in `lab.py`; run `pytest labs/module-03/lesson-01`.
2. **See why the RoPE key is separate:** `python labs/module-03/lesson-01/why_decoupled.py`. Explain each of the three printed errors in one sentence.
3. **Correctness first:** `python labs/module-03/lesson-01/check_mla.py`. Every line must say PASS, including your `absorbed_attention` against the module's naive path.
4. **Measure:**

   ```bash
   python labs/module-03/decode_compare.py --threads 4 --contexts 1024 4096 16384 --rounds 30 --out runs/l31/decode.json
   ```

5. **Report:** the memory table (measured vs formula), the decode-step medians with intervals, the decision the rule gives, and one paragraph on why hypothesis 3 can fail on a CPU and hold on a GPU (use the intensity numbers from Shapes and cost).

Measured in this build (free CPU, Windows 11, 16-thread laptop, torch 2.14.1+cpu, fp32, toy preset, `--threads 4`, 30 rounds; another build job was using the CPU, so intervals are wide): cache bytes equal the formula exactly for every arm and context. At $S = 16{,}384$: Baseline-0 32.5 MiB, GQA $K = 1$ 16.5 MiB, MLA 20.5 MiB (each includes 0.5 MiB of position bookkeeping). Decode-step medians at 16K, with 95% intervals and the paired speed-up against Baseline-0: Baseline-0 14.4 ms [12.7, 15.6]; GQA $K = 1$ 9.6 ms [8.5, 11.8], speed-up 1.37 [1.05, 1.80]; MLA naive 45.1 ms [41.4, 53.8], 0.32 [0.25, 0.35]; MLA absorbed 23.1 ms [20.7, 26.2], 0.61 [0.45, 0.72] (coefficients of variation 35–56%). Decision by the rule: absorbed's interval lies entirely above naive's, so the project uses the absorbed path. Hypothesis 3 is not observed on this CPU: absorbed MLA is about 1.6× slower per step than Baseline-0 at every context measured (1K, 4K, 16K), while caching 37% less — the CPU is not memory-bound at this size. Whether the order flips on an H100 is the pilot's question.

<details>
<summary>Hint for step 1</summary>

`w_kv_b.view(H, d_n + d_v, d_c)` gives each head's rows: the first `d_n` are $W^{UK}_i$, the rest $W^{UV}_i$. For `absorbed_attention`, build `qa = cat(q_lat, q_rope)` and `ka = cat(c[:, None], k_rope)`, call `frontierlab.attention.ops.attend(qa, ka, c[:, None], q_pos, k_pos, scale=scale)` — one key/value head, so it is MQA — then apply $W^{UV}_i$ with an `einsum`.

</details>

<details>
<summary>Hint for step 5</summary>

On a CPU the toy model's decode step is dominated by fixed per-op costs and by arithmetic; reading a few MiB of cache is cheap. Absorbed MLA does $2H(d_c + d_r) + 2Hd_c$ FLOPs per cached token against GQA's $4Hd$. On an H100 both are far below the ridge point, so time follows bytes read, and MLA reads fewer.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-03/lesson-01/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-03/lesson-01`.

</details>

## Common mistakes

- **Caching the up-projected keys and values.** It works, passes every test, and saves nothing: the cache is as large as MHA's. Check `Cache.nbytes()` against $d_c + d_r$ per token and layer.
- **Applying RoPE to $W^{UK} c$.** Training still works; absorption silently becomes wrong, or impossible. The RoPE part must be its own key.
- **Forgetting the scale.** The softmax scale is $1/\sqrt{d_n + d_r}$, the width of the full query–key dot product, not $1/\sqrt{d_c + d_r}$ of the absorbed vectors.
- **Running the naive path at decode.** It re-expands the whole cache every step: 311 GFLOP per token for Baseline-0-sized MLA at 32K, against 5.4 GFLOP absorbed.
- **Comparing MLA only against MHA.** Your baseline already has grouped KV heads; compare with a KV-matched GQA as well.
- **Trusting a float64 equivalence check on a freshly initialised model.** Near-uniform attention and tiny layer outputs hide real errors; sharpen the weights and check that a planted bug fails.

## References

- DeepSeek-AI, *DeepSeek-V2*, sections 2.1.2–2.1.3, Table 1, abstract. https://arxiv.org/abs/2405.04434
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, section 2.1.1. https://arxiv.org/abs/2412.19437
- Moonshot AI, *Kimi K2*, section 2.1 (QK-Clip and the shared rotary key). https://arxiv.org/abs/2507.20534
- N. Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* (MQA). https://arxiv.org/abs/1911.02150
- J. Ainslie et al., *GQA*. https://arxiv.org/abs/2305.13245
- Hugging Face Transformers, `modeling_deepseek_v3.py`, main branch (checked 2026-10-03). https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py
- Shared code: `labs/common/frontierlab/attention/mla.py`, `checks.py`, `bench.py`, `accounting.py`; tests in `labs/common/tests/test_attention_m03.py`. Versions: [references/versions.md](../../references/versions.md).

## Next

[03.2 · Local/global attention and KV arithmetic](lesson-02.md)
