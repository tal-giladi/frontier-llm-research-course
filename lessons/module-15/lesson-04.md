---
id: "15.4"
module: 15
minutes: 30
practice_minutes: 60
prerequisites: ["15.3", "08.1"]
objectives:
  - Implement asymmetric group quantisation and KIVI's layout (keys per channel, values per token, a full-precision residual window) and verify them against a reference.
  - Measure what 8-, 4-, 3- and 2-bit KV caches and round-to-nearest weight formats do to next-token loss and KL with paired intervals, decoding with the cache as a server does.
  - Convert a KV or weight format into bytes per sequence and batch capacity for the Stage D model at 128K, and say when each format is worth its quality cost.
  - Map the formats onto vLLM 0.30.0's options and know where the inference course takes weight quantisation further.
volatility: implementation
sources:
  - title: "Liu et al. — KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache (sections 3.3, 4.1; abstract)"
    url: https://arxiv.org/abs/2402.02750
  - title: "Hooper et al. — KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization"
    url: https://arxiv.org/abs/2401.18079
  - title: "Frantar et al. — GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
    url: https://arxiv.org/abs/2210.17323
  - title: "Lin et al. — AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
    url: https://arxiv.org/abs/2306.00978
  - title: "Xiao et al. — SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"
    url: https://arxiv.org/abs/2211.10438
  - title: "vLLM v0.30.0 documentation — Quantized KV cache"
    url: https://docs.vllm.ai/en/v0.30.0/features/quantization/quantized_kvcache/
  - title: "LLM Inference and Deployment (sibling course), Module 4: Quantization"
    url: https://github.com/tal-giladi/ai-inference-course/blob/main/lessons/module-04/lesson-01.md
last_verified: "2026-10-07"
---

# 15.4 · KV and weight quantisation for serving

Extension: lesson 15.3 showed that at long context the KV cache, not the weights, decides how many sequences a GPU can hold; this lesson shrinks both. You implement asymmetric group quantisation and KIVI's cache layout, plug the quantised cache into Baseline-0's attention as a new kind, and measure what 8, 4, 3 and 2 bits per element do to next-token loss when the model decodes with that cache, then do the same for round-to-nearest weight formats. The arithmetic turns each format into sequences per GPU for the Stage D model. Weight quantisation methods (GPTQ, AWQ, GGUF) and their tooling are taught in the sibling course, LLM Inference and Deployment, Module 4; here they appear only where they change a capacity decision.

## Why this matters at a frontier lab

At 128K tokens the Stage D model's BF16 cache is 14 GiB per sequence and its weights 3.2 GiB: halving the weights buys almost nothing, halving the cache doubles capacity. For reasoning RL and long agent episodes that capacity is throughput (lesson 15.3). KV quantisation is also the cheapest of the memory levers: it needs no retraining, unlike MLA or windows, and it can be switched on per deployment. The risk is quality that degrades only at long context or only on hard prompts, which is why the measurement has to decode with the quantised cache, not quantise a tensor and report its error.

## The idea

### Asymmetric group quantisation

A group of $G$ values $x_1 \ldots x_G$ is stored as $b$-bit integers with a scale and a zero point:

$$s = \frac{\max_i x_i - \min_i x_i}{2^b - 1}, \qquad q_i = \text{clamp}\Big(\text{round}\Big(\frac{x_i - \min_j x_j}{s}\Big), 0, 2^b - 1\Big), \qquad \hat x_i = s\,q_i + \min_j x_j.$$

With a 16-bit scale and a 16-bit zero point per group, the storage is $b + 32/G$ bits per element. The rounding error is at most $s/2$. Symmetric formats (Module 8) map $[-\max|x|, \max|x|]$ onto the grid and waste half of it when a group is not centred on zero; the zero point removes that waste for the price of 16 bits per group.

### Which axis to group along

Attention reads keys through $q^\top k$: an error in a key channel that is large *for every token* shifts every score of every query. KIVI (sections 3–4) observes that key caches have a few such outlier channels and value caches do not, and quantises **keys per channel** (each channel's values over $G$ consecutive tokens share a scale, so an outlier channel gets its own large scale) and **values per token** (each token's $d$ values share one). Because per-channel groups need $G$ tokens before they can be quantised, the most recent tokens stay in full precision: a **residual window** of $R$ tokens, which also protects the tokens attention usually weights most. KIVI uses $G = 32$, $R = 128$ (section 4.1). KVQuant adds quantising keys *before* RoPE (the rotation mixes channel pairs and smears outliers), non-uniform levels and a sparse set of outliers kept exact.

### Weights

Weights are quantised once, offline. Round-to-nearest (RTN) per output row in groups of $G$ input channels, asymmetric, at 4 bits with $G = 128$ ("w4a16 g128") is the usual baseline. GPTQ improves on it by quantising one column at a time and updating the remaining columns to compensate, using second-order information from calibration data; it quantises 175B models to 3–4 bits in about 4 GPU-hours. AWQ observes that protecting about 1% of salient weight channels, chosen by activation magnitude, recovers most of the loss, and protects them by an equivalent per-channel scaling rather than mixed precision. SmoothQuant moves quantisation difficulty from activations to weights with the same kind of equivalent scaling, for W8A8 inference. The inference course's Module 4 covers these methods, GGUF and the conversion tools in depth.

### What a format buys

Cache bytes of one sequence at context $S$ with $e$ elements per token per layer, $L$ layers, KIVI layout:

$$\text{KV}(S) = L\Big[n_q\, e\,\Big(b + \frac{32}{G}\Big) + (S - n_q)\, e \cdot 16\Big]\Big/8, \qquad n_q = G\Big\lfloor \frac{S - R}{G} \Big\rfloor.$$

Then lesson 15.3's capacity formula applies unchanged.

## Worked example

**One group, 2 bits.** $x = (0.0, 0.4, 0.6, 3.0)$, $b = 2$: $s = 3.0/3 = 1.0$, $q = \text{round}(0, 0.4, 0.6, 3.0) = (0, 0, 1, 3)$, $\hat x = (0, 0, 1, 3)$. The error is at most $s/2 = 0.5$ (here 0.4). With $b = 1$: $s = 3$, $q = (0, 0, 0, 1)$, $\hat x = (0, 0, 0, 3)$.

**Why per channel for keys.** Two tokens, three channels, channel 1 an outlier: $k_1 = (8.0, 0.1, -0.1)$, $k_2 = (8.2, -0.1, 0.1)$, 2 bits. Per token, token 1's group spans $[-0.1, 8.0]$, so $s = 8.1/3 = 2.7$, and $0.1$ maps to $\text{round}(0.2/2.7) = 0$, back to $-0.1$. Channels 2 and 3 of both tokens become $-0.1$ and $-0.1$: their sign, the only information they had, is gone. Per channel (over the two tokens), channel 1 spans $[8.0, 8.2]$ with $s = 0.067$, and channels 2 and 3 span $[-0.1, 0.1]$ with $s = 0.067$: every value is kept to within $0.033$.

**Bytes at 128K for the Stage D model.** $e = 2 \cdot 8 \cdot 128 = 2{,}048$, $L = 28$. BF16: $28 \cdot 131{,}072 \cdot 2{,}048 \cdot 2 = 14.0$ GiB. KIVI-2 ($b = 2$, $G = 32$, $R = 128$): $n_q = 131{,}008$ tokens at 3 bits per element, 64 at 16: $28 \cdot (131{,}008 \cdot 2{,}048 \cdot 3 + 64 \cdot 2{,}048 \cdot 16)/8 = 2.64$ GiB. Capacity on 80 GB next to the BF16 weights: from 4 sequences to 26.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| cached keys, values per layer (emulation keeps full precision) | (B, KV, S, d) | float32 (toy) / bf16 | CPU / GPU |
| keys seen by attention | (B, KV, S, d), first $n_q$ tokens through `asym_qdq` along dim 2 | same | same |
| values seen by attention | (B, KV, S, d), first $n_q$ tokens through `asym_qdq` along dim 3 | same | same |
| per-group scale and zero point (a real cache) | keys (B, KV, n_q/G, d); values (B, KV, n_q, d/G) | fp16 | same |
| quantised weights (emulated) | (out, in), groups of G along `in` | float32 holding grid values | same |

Emulation cost: one extra quantise-dequantise of the older part of the cache per decode step, which makes the lab slower, not faster. A real kernel reads the packed integers and dequantises in registers, and the speed-up comes from reading fewer bytes. That needs a fused kernel, which is the serving engine's job.

## Build it

```python
from frontierlab.ttc import kvquant as KQ
KQ.asym_qdq(x, bits=4, dim=1, group=32)
k_q, v_q = KQ.kivi_qdq(k, v, bits=2, group=32, residual=128)          # keys per channel, values per token
model_q = KQ.with_quant_kv(model, bits=2, group=32, residual=32)      # attention kind "gqa-kvq", same weights
KQ.decode_nll(model_q, windows, prefill=1, ref=model)                 # one-token steps with the cache; NLL and KL
KQ.quantize_weights_(copy_of_model, bits=4, group=128)                # RTN on every block linear; embedding/head kept
```

`frontierlab/ttc/kvquant.py` registers `"gqa-kvq"`: Baseline-0's GQA whose attention, whenever a cache is used, reads the cache through `kivi_qdq`. Groups are aligned to absolute positions, so a token's quantised value never changes once its group has left the residual window, which is what a quantise-once cache does. Correctness checks (`test_ttc.py`): the rounding error is within $s/2$ for 2, 4 and 8 bits; constant groups are exact; the residual window is untouched; at most $2^b$ distinct values per channel group; at 16 bits the quantised-cache model matches the original (KL below $10^{-8}$); no quantisation while the context fits in the window; RTN leaves the embedding and tied head unchanged and reports bits per weight.

## What the evidence says

- **FP8 KV cache: ESTABLISHED in engines.** vLLM 0.30.0's `--kv-cache-dtype` accepts `auto`, `fp8` (E4M3), `fp8_e4m3`, `fp8_e5m2` and others; scales come from the checkpoint (`k_scale`, `v_scale`) or default to 1.0, and llm-compressor can calibrate them (quantized KV-cache docs at the tag). Older versions documented an on-the-fly `calculate_kv_scales`; the 0.30.0 page does not, so check before relying on it.
- **2-bit asymmetric KV (KIVI): PROMISING.** KIVI reports 2.6× less peak memory including weights, up to 4× larger batch and 2.35×–3.47× throughput on real workloads (abstract), with its own kernels. KVQuant reports under 0.1 perplexity degradation at 3 bits and a 1M-token context for LLaMA-7B on one A100-80GB (abstract). Engine support for sub-8-bit KV varies by version; check the tag.
- **Weight-only INT4 (GPTQ, AWQ): ESTABLISHED.** GPTQ: 3–4 bits, 175B in about 4 GPU-hours, about 3.25× faster generation on an A100. AWQ: protecting about 1% salient weights. vLLM 0.30.0 `--quantization` covers `awq`, `awq_marlin`, `gptq`, `gptq_marlin`, `fp8`, `compressed-tensors`, `modelopt_fp4`, `mxfp4` and more. SmoothQuant (W8A8): up to 1.56× speed-up and 2× memory reduction (abstract).
- **Course measurement (free CPU, 2026-10-07; the toy target of lesson 15.2, emulation):** KV at 8 and 4 bits changed next-token loss by at most +0.0005 nats; 3 bits by +0.0011 to +0.0016; 2 bits by about +0.01. Keys per channel beat keys per token at 4 bits (+0.0003 against +0.0005), and a 32-token residual window halved the KL at 3 and 2 bits. The toy's key outliers are mild (largest channel about 2–2.5× the median), so the per-channel advantage is far smaller than in large models. Weights: INT8 was free, INT4 g128 cost +0.0018 asymmetric and +0.0029 symmetric, INT2 +0.040. These are a 2M-parameter model's numbers, not a prediction for Qwen3.

## Lab

**Folder:** [`labs/module-15/lesson-04/`](../../labs/module-15/) · **Time:** about 60 minutes (about 2 minutes unattended after lab 15.2) · **Pass check:** `pytest labs/module-15/lesson-04` passes; `quant_lab.py` prints all five parts; your write-up picks a KV format and a weight format for 128K serving of the Stage D model and justifies both with the measured quality cost and the computed capacity.

### Experiment contract

- **Question:** how much next-token quality does each KV and weight format cost, decoded the way a server decodes, and how much capacity does it buy at 128K? Decision informed: the cache and weight formats for long-context serving and RL rollouts of the Stage D model.
- **Hypothesis:** 8- and 4-bit KV are near-lossless, 2-bit costs visibly; keys per channel beat keys per token; the residual window matters most at 2 bits; INT4 weights cost little, INT2 a lot. Status: reported (KIVI, KVQuant, GPTQ, AWQ); the per-channel advantage may be small in a toy model without strong outliers.
- **Baseline:** the same model with a full-precision cache and weights, decoded with the same code.
- **Changed variable:** bits, key grouping axis and residual window (KV); bits, group size and symmetry (weights). **Controlled:** the 15.2 target checkpoint, 48 fixed validation windows of 256 tokens (KV) and 64 (weights), group size 32 for KV, float32 emulation.
- **Comparison axis:** equal model and data; storage bits per element, scales included.
- **Budget:** free CPU, about 2 minutes; main path below 1 GPU-hour (PROJECTED).
- **Metrics and decision rule:** per-window mean NLL difference with a paired 95% bootstrap interval, and mean KL to the full-precision next-token distribution. A format is "free" at this scale if the interval includes 0 or lies below +0.001 nats.
- **Correctness checks:** `pytest labs/module-15/lesson-04`; your functions equal the course's (checked by the script); the 16-bit and in-window identities in `test_ttc.py`.
- **Fallback evidence:** KIVI's and KVQuant's published tables, labelled as published.
- **Limits:** emulation, so no speed; a 2M-parameter model trained for 800 steps; mild outliers; short windows, so the residual window covers a large share of each window; no long-context task evaluation (Module 4's Eval v1 is the place for that on the main path).

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 80 GB, vLLM 0.30.0. Not run in this build; part of the Module 15 pilot | `--variant main --print`: `vllm bench throughput` with `--kv-cache-dtype auto` and `fp8` at 8K-token outputs, then Eval v2 (GSM8K strict, LAMBADA) on the model served with each; INT4 weights through an AWQ or GPTQ checkpoint as in the inference course's Module 4. **PROJECTED:** under 1 GPU-hour |
| Free GPU (Colab/Kaggle T4) | T4 | the T4 has no FP8 tensor cores; run this lab's emulation on the GPU with a larger model (Qwen3-0.6B-Base through `frontierlab.pipeline.hf_eval`) |
| Free CPU | laptop; measured 103 s after lab 15.2's models exist (other jobs running) | the steps below |

### Steps

1. **Implement** the three TODOs in `lab.py` and run `pytest labs/module-15/lesson-04`.
2. **Run** `python labs/module-15/lesson-04/quant_lab.py`. Read the outlier table first: how much larger is the largest key channel than the median, in each layer?
3. **KV.** Rank the formats by bits per element and by NLL change. Where does the residual window matter, and why more at 2 bits than at 4?
4. **Weights.** Compare g128 with g32 at 4 bits, and asymmetric with symmetric. Which difference is outside its interval?
5. **Decide.** With the capacity table, choose a KV format and a weight format for the Stage D model serving 128K contexts, and the one evaluation you would run on the main path before shipping it.

<details>
<summary>Hint for TODO 2</summary>

`n_q = max(0, (S - residual) // group) * group`. Quantise `k[:, :, :n_q]` with `asym_qdq(..., dim=2, group=group)` and `v[:, :, :n_q]` with `dim=3`, then concatenate the untouched rest along dim 2.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-15/lesson-04/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-15/lesson-04`. The build's measured numbers are in "What the evidence says" above.

</details>

## Common mistakes

- **Reporting tensor error instead of decoding quality.** A 1% relative error in the keys can change attention a lot or not at all; decode with the quantised cache and measure the loss.
- **Quantising the newest tokens.** They carry most of the attention mass; keep a residual window in full precision.
- **Grouping keys per token.** Outlier channels then set the scale of every group and the small channels are lost.
- **Counting only the element bits.** Scales and zero points cost 32/G bits per element: 1 bit per element at G = 32.
- **Quantising the embedding or the tied output head with the rest.** Most recipes keep them in higher precision; the lab does too.
- **Assuming an emulated format is faster.** Speed needs kernels that read packed integers; emulation only tells you the quality.

## References

- Z. Liu et al., *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*, 2024, sections 3.3 and 4.1. https://arxiv.org/abs/2402.02750
- C. Hooper et al., *KVQuant*, 2024. https://arxiv.org/abs/2401.18079
- E. Frantar et al., *GPTQ*, 2022. https://arxiv.org/abs/2210.17323
- J. Lin et al., *AWQ*, 2023. https://arxiv.org/abs/2306.00978
- G. Xiao et al., *SmoothQuant*, 2022. https://arxiv.org/abs/2211.10438
- vLLM v0.30.0, *Quantized KV cache*. https://docs.vllm.ai/en/v0.30.0/features/quantization/quantized_kvcache/
- Sibling course, *LLM Inference and Deployment*, Module 4 (Quantization): [lesson 4.1](https://github.com/tal-giladi/ai-inference-course/blob/main/lessons/module-04/lesson-01.md), [4.2](https://github.com/tal-giladi/ai-inference-course/blob/main/lessons/module-04/lesson-02.md), [4.3](https://github.com/tal-giladi/ai-inference-course/blob/main/lessons/module-04/lesson-03.md).
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

The [Module 15 project](../../projects/module-15-ttc-report.md) puts the module together: a budget-matched test-time-compute report with a recommendation for a stated latency target. Module 16 then trains agents, whose episodes are long rollouts of exactly the kind this module priced.
