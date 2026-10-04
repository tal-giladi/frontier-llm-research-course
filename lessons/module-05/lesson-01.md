---
id: "05.1"
module: 5
minutes: 40
practice_minutes: 90
prerequisites: ["03.2", "04.1", "01.4", "02.4"]
objectives:
  - Derive the gated delta rule from linear attention, and state what the decay and the delta term each add, with a hand-worked two-token example.
  - Implement the recurrent and the chunked-parallel forms of channel-gated delta attention and show by test that they agree to float64 rounding for every chunk size.
  - Compute the decode memory of a 3:1 linear/full hybrid from its config (Baseline-0, Qwen3-Next, Kimi Linear) and explain why the saving approaches but never exceeds the share of linear layers.
  - Run an equal-parameter, equal-token comparison of two hybrids against Baseline-0 under an experiment contract and report the result with paired intervals.
volatility: concept
sources:
  - title: "Kimi Linear: An Expressive, Efficient Attention Architecture (KDA recurrence, channel-wise gate, chunkwise algorithm with chunk size 64, uniform 3:1 KDA:MLA, NoPE MLA, KV and decode claims)"
    url: https://arxiv.org/abs/2510.26692
  - title: "Yang, Kautz, Hatamizadeh — Gated Delta Networks: Improving Mamba2 with Delta Rule (gated delta rule, WY chunkwise form, hybrids with sliding-window attention)"
    url: https://arxiv.org/abs/2412.06464
  - title: "Yang et al. — Parallelizing Linear Transformers with the Delta Rule over Sequence Length (WY representation for chunked DeltaNet)"
    url: https://arxiv.org/abs/2406.06484
  - title: "Qwen3-Next-80B-A3B-Instruct model card and config.json (12 x (3 Gated DeltaNet + 1 Gated Attention); full_attention_interval 4; 10% of Qwen3-32B training cost)"
    url: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
  - title: "Kimi-Linear-48B-A3B-Instruct config.json (linear_attn_config.full_attn_layers)"
    url: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct
  - title: "flash-linear-attention 0.5.2 (fla.ops.kda.chunk_kda, fla.ops.gated_delta_rule.chunk_gated_delta_rule)"
    url: https://github.com/fla-org/flash-linear-attention
  - title: "Why Did MiniMax M2 End Up as a Full Attention Model? (MiniMax, Hugging Face blog)"
    url: https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model
last_verified: "2026-10-04"
---

# 05.1 · Linear and hybrid attention at scale

Full attention keeps every past token and pays for it twice: compute that grows with the square of the context in training, and a KV cache that grows with the context at decode. Linear attention replaces the cache with a fixed-size matrix, a memory that is updated once per token. This lesson takes the linear attention of the parent course (lesson 12.5) to the form that 2025 frontier models actually ship: the gated delta rule of Gated DeltaNet (Qwen3-Next) and its channel-wise variant, Kimi Delta Attention (Kimi Linear). You will build both the token-by-token recurrence and the chunked-parallel form that makes training fast, prove they compute the same function, work out what a 3:1 hybrid saves, and run a first controlled comparison against Baseline-0.

## Why this matters at a frontier lab

At 1M tokens Baseline-0's KV cache is 12 GiB per sequence; a 3:1 hybrid caches 3 GiB, and its linear layers cost the same per token at 1K and at 1M. That is why two labs with very different stacks, Moonshot and Alibaba's Qwen team, published 3:1 hybrids in the same autumn of 2025, and why MiniMax, after trying hybrids, published an explanation of why its M2 model uses full attention everywhere. The decision is not "is linear attention faster" — per token at long context it plainly is — but whether the quality it gives up shows up in the evaluations you trust, whether your kernels and serving stack support it, and at which context length the saving starts to matter. Getting there needs three things this lesson provides: a correct implementation (the chunked form is easy to get subtly wrong), an exact cost model, and a controlled comparison.

## The idea

### From linear attention to a memory you can edit

Softmax attention for query $q_t$ reads every earlier key and value. Linear attention (parent lesson 12.5) drops the softmax and keeps one matrix per head,

$$S_t = S_{t-1} + k_t v_t^\top, \qquad o_t = S_t^\top q_t,$$

with $S_t \in \mathbb{R}^{d_k \times d_v}$ the state, $k_t, q_t \in \mathbb{R}^{d_k}$, $v_t \in \mathbb{R}^{d_v}$. Decoding costs $O(d_k d_v)$ per token whatever the context. Two weaknesses limited it: nothing is ever forgotten, so old associations pile up and interfere, and writing a value for a key that is already stored *adds* to the old value instead of replacing it.

**Gating** fixes the first: multiply the state by a data-dependent decay before writing, $S_t = \alpha_t S_{t-1} + k_t v_t^\top$ with $\alpha_t \in (0, 1)$ (Mamba-2, GLA).

**The delta rule** fixes the second. Before writing, ask the memory what it currently returns for $k_t$, $\hat v_t = S_{t-1}^\top k_t$, and move it a fraction $\beta_t \in (0, 1)$ of the way to the target: $S_t = S_{t-1} + \beta_t k_t (v_t - \hat v_t)^\top$. With a unit-norm key and $\beta_t = 1$, the old value stored under $k_t$ is replaced exactly. This is one step of online least-squares regression of values on keys.

**The gated delta rule** (Gated DeltaNet, arXiv 2412.06464) does both. **Kimi Delta Attention** (KDA, arXiv 2510.26692) makes the decay a vector, one forgetting rate per key channel:

$$S_t = (I - \beta_t k_t k_t^\top)\,\mathrm{Diag}(\alpha_t)\, S_{t-1} + \beta_t k_t v_t^\top, \qquad o_t = S_t^\top q_t,$$

where $\alpha_t \in [0, 1]^{d_k}$ is the per-channel decay and $\beta_t \in [0, 1]$ the write strength. Gated DeltaNet is the special case $\mathrm{Diag}(\alpha_t) = \alpha_t I$ with one scalar per head. The Kimi Linear paper describes the difference in one line: Gated DeltaNet has a single scalar gate per head, KDA a "channel-wise variant in which each feature dimension maintains an independent forgetting rate". Different channels can then hold information for different horizons, a little like RoPE's frequencies.

The layer around the recurrence (both papers, in outline): $q, k, v$ from linear projections, a short causal depthwise convolution (kernel 4) and SiLU; $q$ and $k$ L2-normalised (which keeps $I - \beta k k^\top$ a contraction); $\beta_t = \sigma(W_\beta x_t)$; $\log \alpha_t = -e^{A} \cdot \mathrm{softplus}(\cdot)$ of a projection of $x_t$ (low-rank in KDA); the head outputs RMS-normalised and multiplied by a data-dependent gate before the output projection. There is no RoPE in these layers: order comes from the decay and the convolution.

### Why a chunked form is needed

The recurrence is sequential: token $t$ needs $S_{t-1}$. On a GPU that means $T$ small dependent steps, each far too small to use the tensor cores. The chunked-parallel form processes $C$ tokens at a time. Inside a chunk with start state $S_0$, write $b_r = \sum_{i \le r} \log \alpha_i$ (per channel), $\Gamma_r = e^{b_r}$, and $e_r = \beta_r (v_r - (\mathrm{Diag}(\alpha_r) S_{r-1})^\top k_r)$, the actual correction written at step $r$. Unrolling the recurrence gives

$$S_r = \mathrm{Diag}(\Gamma_r) S_0 + \sum_{i \le r} \mathrm{Diag}(\Gamma_r / \Gamma_i)\, k_i e_i^\top,$$

and substituting it into the definition of $e_r$ gives one linear system for all $C$ corrections at once:

$$(I + A)\,E = \mathrm{Diag}(\beta)\,(V - \tilde K S_0), \qquad A_{ri} = \beta_r\, k_r^\top \mathrm{Diag}(\Gamma_r/\Gamma_i)\, k_i \ \ (i < r),$$

with $E$ the $C \times d_v$ matrix of corrections and $\tilde K$ the rows $\Gamma_r \odot k_r$. $I + A$ is unit lower-triangular, so $E = U - W S_0$ with $U = (I+A)^{-1}\mathrm{Diag}(\beta) V$ and $W = (I+A)^{-1}\mathrm{Diag}(\beta)\tilde K$: the "WY" or UT-transform form of DeltaNet (Yang et al., arXiv 2406.06484), Gated DeltaNet and KDA. Then

$$O = \tilde Q S_0 + M E, \quad M_{ri} = q_r^\top \mathrm{Diag}(\Gamma_r/\Gamma_i)\, k_i \ (i \le r), \qquad S_C = \mathrm{Diag}(\Gamma_C) S_0 + \hat K^\top E,$$

with $\tilde Q$ the rows $\Gamma_r \odot q_r$ and $\hat K$ the rows $(\Gamma_C/\Gamma_i) \odot k_i$. Every step is a matrix product or a triangular solve; only the $T/C$ chunk boundaries are sequential. This is not an approximation: the chunked form must reproduce the recurrence to rounding error, and that equality is the "recurrent versus parallel equivalence" check of the course correctness suite.

### Hybrids: a few full layers to do what the state cannot

A $d_k \times d_v$ state can hold at most $d_k$ exactly retrievable associations (that many orthogonal keys). Full attention has no such limit. Both shipped designs therefore keep some full-attention layers:

| Model | Layout (PUBLICLY DOCUMENTED) | Linear layers | Full layers |
|---|---|---|---|
| Qwen3-Next-80B-A3B | "12 * (3 * (Gated DeltaNet -> MoE) -> 1 * (Gated Attention -> MoE))" (model card); `full_attention_interval` 4 | 36 Gated DeltaNet, 16 QK heads and 32 V heads of width 128 | 12 gated attention, 16 Q / 2 KV heads of width 256 |
| Kimi Linear 48B-A3B | "a uniform 3:1 ratio" of KDA to MLA (paper); `full_attn_layers` = 4, 8, ..., 24, 27 of 27 (config) | 20 KDA, 32 heads of width 128 | 7 MLA without positional encoding (`mla_use_nope`) |

Note the config detail: Kimi Linear's last block is 2:1, so 7 of 27 layers are full, not 6.75.

## Worked example

### Two tokens by hand

$d_k = 2$, $d_v = 1$, $S_0 = 0$. Token 1: $k_1 = (1, 0)$, $v_1 = 2$, $\beta_1 = 1$, $\alpha_1 = (1, 1)$. Token 2: $k_2 = (0.6, 0.8)$ (unit norm), $v_2 = 1$, $\beta_2 = 0.5$, $\alpha_2 = (0.5, 1)$: channel 0 forgets half, channel 1 nothing. Query at step 2: $q_2 = (1, 0)$.

Recurrent. Step 1: $\hat v_1 = 0$, so $S_1 = k_1 \cdot 1 \cdot 2 = (2, 0)^\top$. Step 2: decay, $\mathrm{Diag}(\alpha_2) S_1 = (1, 0)^\top$; prediction $\hat v_2 = 0.6 \cdot 1 + 0.8 \cdot 0 = 0.6$; correction $e_2 = 0.5\,(1 - 0.6) = 0.2$; $S_2 = (1, 0)^\top + 0.2\,(0.6, 0.8)^\top = (1.12, 0.16)^\top$. Output $o_2 = S_2^\top q_2 = 1.12$.

Chunked, both tokens in one chunk. $b_1 = (0, 0)$, $b_2 = (\ln 0.5, 0)$, so $\Gamma_2/\Gamma_1 = (0.5, 1)$. $A_{21} = \beta_2\, k_2^\top \mathrm{Diag}(0.5, 1)\, k_1 = 0.5 \cdot (0.6 \cdot 0.5 \cdot 1 + 0.8 \cdot 1 \cdot 0) = 0.15$. With $S_0 = 0$ the system is $e_1 = \beta_1 v_1 = 2$ and $e_2 + 0.15\, e_1 = \beta_2 v_2 = 0.5$, so $e_2 = 0.2$ — the same correction as the recurrence. $M_{21} = q_2^\top \mathrm{Diag}(0.5, 1)\, k_1 = 0.5$, $M_{22} = q_2^\top k_2 = 0.6$, so $o_2 = 0.5 \cdot 2 + 0.6 \cdot 0.2 = 1.12$. Same answer.

Plain linear attention for comparison: $S_2 = k_1 v_1 + k_2 v_2 = (2.6, 0.8)^\top$ and $o_2 = 2.6$. It returns the old value plus interference from $k_2$; the gated delta rule returned a decayed old value plus a small, *corrective* write.

### Decode memory of a 3:1 hybrid

Baseline-0 shape, BF16 K/V, fp32 state (what our implementation keeps; see Shapes and cost). A full layer caches $2 \cdot 4 \cdot 64 \cdot 2 = 1{,}024$ bytes per token. A linear layer holds $12 \cdot 64 \cdot 64 \cdot 4 = 196{,}608$ bytes of state plus a convolution tail of $3 \cdot 12 \cdot 192 \cdot 2 = 13{,}824$ bytes: 205.5 KiB, whatever the context. Layout `LLLFLLLFLLLF`:

$$\text{bytes}(S) = 3 \cdot 1{,}024 \cdot S + 9 \cdot 210{,}432$$

At $S = 32{,}768$: $96$ MiB $+ 1.81$ MiB $= 97.8$ MiB against 384 MiB for Baseline-0, a ratio of 0.255. At 1M the ratio is 0.250: the saving approaches the share of linear layers, 9/12, and can never exceed it. For Kimi Linear the same arithmetic (7 MLA layers caching $512 + 64$ latent elements per token, 20 KDA states of $32 \cdot 128 \cdot 128$) gives a 73.0% reduction against an all-MLA model at 128K and 73.9% at 1M (`equivalence.py`, part 2); the paper's "up to 75%" is the limit of an exact 3:1 ratio.

## Shapes and cost

| Tensor (one KDA layer, Baseline-0 width) | Shape | dtype / device |
|---|---|---|
| input $x$ | (B, T, 768) | bf16 under autocast on GPU, fp32 on CPU |
| $q, k, v$ after conv, norm | (B, 12, T, 64) | same |
| log-decay $g$ | (B, 12, T, 64) — (B, 12, T) expanded for Gated DeltaNet | fp32 |
| $\beta$ | (B, 12, T) | same as $x$, cast to fp32 in the core |
| pairwise decay inside a chunk (our reference) | (B, 12, C, C, 64) | fp32 |
| state $S$ in `LayerCache["state"]` | (B, 12, 64, 64) | fp32, also under bf16 autocast |
| conv tail in `LayerCache["conv"]` | (B, 3, 12 · 192) | same as $x$ |

**Mixing FLOPs per token and head** (forward; projections are in $2N$ as for every kind): recurrent $\approx 7 d_k d_v$ (decay, $S^\top k$, rank-1 update, $S^\top q$); chunked $\approx 6 d_k d_v + C(4.5 d_k + 2 d_v)$ in our implementation (`accounting.linear_mix_flops`). A full GQA layer costs $4 H d$ per (query, key) pair, $2 H d T$ per token on average in training. Setting one equal to the other for Baseline-0's shape with $C = 64$: $12 \cdot (6 \cdot 64^2 + 64 \cdot 416) / (2 \cdot 12 \cdot 64) = 412$ tokens (`accounting.linear_crossover_length`). Below about 400 tokens of training context a linear layer does *more* arithmetic than the full layer it replaces; above it, less, and at 32K about 80 times less. FLOPs are not time: lesson 05.3 measures what the kernels make of it.

**Parameters.** A KDA layer here has $q, k, v, g$ and output projections ($5 \cdot 768^2$) plus small gate and decay terms, about 1.4M more than a GQA layer with 4 KV heads. A 3:1 hybrid of Baseline-0 has 109.8M non-embedding parameters instead of 96.8M; the equal-parameters arm shrinks the SwiGLU width from 2,816 to 2,346 (`accounting.match_intermediate`).

**Precision of the state.** Our layer keeps the state in fp32 even under bf16 autocast: a state that is decayed and corrected millions of times accumulates rounding error. Storing it in lower precision is one of the open infrastructure problems MiniMax lists ("low-precision state storage"); serving stacks that keep it in bf16 should be checked against an fp32 reference on long sequences.

## Build it

`labs/common/frontierlab/attention/deltanet.py` registers `"gdn"` and `"kda"`; `hybrid.py` registers `"hybrid"`, which takes `layer_idx` and builds a linear or a full kind per layer:

```python
from frontierlab.attention import deltanet, hybrid            # registers gdn, kda, hybrid
from frontierlab.model import LM, baseline0

cfg = baseline0().with_(attention="hybrid", extra={"full_every": 4, "linear_kind": "kda", "full_kind": "gqa"})
hybrid.hybrid_layer_types(cfg)       # ['linear', 'linear', 'linear', 'full', ...]: Qwen3-Next's interval 4
qwen_next_like = cfg.with_(extra={**cfg.extra, "linear_kind": "gdn", "full_kind": "gated"})
kimi_like = cfg.with_(num_hidden_layers=27,
                      extra={**cfg.extra, "layer_types": hybrid.from_full_layer_list([4, 8, 12, 16, 20, 24, 27], 27)})
```

The core is two functions with the same signature, `delta_rule_recurrent(q, k, v, g, beta, state)` and `delta_rule_chunked(..., chunk)`, each returning the outputs and the final state, so decoding is "chunked prefill, then one recurrent step per token, carrying the state in the `LayerCache`". `linear_mode` selects `"chunked"`, `"recurrent"` or `"fla"`: flash-linear-attention 0.5.2's `chunk_kda` and `chunk_gated_delta_rule` (layout `[B, T, H, D]`, log-space gate, `scale=1.0` because our queries are pre-scaled). The `"fla"` path is **not run in this build**; it is part of the Module 5 pilot, whose first step is `test_fla_matches_reference` in `labs/common/tests/test_attention_m05.py` (skipped without CUDA).

The pairwise decay tensor $\exp(b_r - b_i)$ is formed with the exponent masked to $-\infty$ above the diagonal *before* `exp`: above the diagonal $b_r - b_i > 0$ and grows with the chunk, and $e^{600}$ is infinity in any float format. fla avoids the $C \times C \times d_k$ tensor with a factorised form and safeguards on the gate; our reference trades memory for exactness at any decay.

Correctness, measured in this build (float64, RMSNorm in the input dtype, sharpened weights; `pytest labs/common/tests/test_attention_m05.py`: 60 passed and 2 GPU-only tests skipped, about 50 s): op-level gradcheck of every kind with respect to its input and every parameter; causal check differences of exactly 0; cached decode in 1-, 3- and 7-token steps after a 6-token prefill within $3 \times 10^{-15}$; recurrent against chunked at chunk sizes 1, 2, 4, 7, 16 and 64 within $2 \times 10^{-15}$ for outputs and final states, including cumulative decays down to $e^{-50}$ inside one chunk and the gradients of both forms; cache bytes equal to the formula. One more test pins down the delta rule itself: with orthonormal keys, writing the same key twice leaves only the second value (5, 5), where plain linear attention returns their sum (6, 5).

## What the evidence says

- **Linear/full hybrids: PROMISING.** Two labs ship 3:1 hybrids at scale with documented layouts (table above), and Gated DeltaNet's paper reports hybrids with sliding-window attention; independent replication of the quality claims at frontier scale is limited, and MiniMax reports the opposite experience (below).
- **KDA's channel-wise gate: MODEL-SPECIFIC.** Kimi Linear reports it against Gated DeltaNet hybrids and full MLA trained with "identical training recipe" on 1.4T tokens (paper). The claims "reducing KV cache usage by up to 75%" and "up to 6× decoding throughput for a 1M context" are Moonshot's measurements on its own stack (company claim); the 75% is the arithmetic limit of a 3:1 layout, as the worked example shows.
- **Qwen3-Next:** the model card says it "outperforms Qwen3-32B-Base on downstream tasks with 10% of the total training cost" and has "10 times inference throughput for context over 32K tokens" (company claims; the training-cost comparison mixes an MoE with a dense model, so it is not an attention ablation).
- **The counter-case (company claim).** MiniMax writes that a lightning-attention hybrid "looked just as good as pure full attention" on standard benchmarks, but at larger scale "the model had clear deficits in complex, multi-hop reasoning tasks", and that low-precision state storage, prefix caching and speculative decoding were unsolved for linear attention in their stack. Lesson 05.3 turns this into a testable prediction for Eval v1.
- **Equivalence of the chunked form: ESTABLISHED** mathematics (it is an exact rewriting), and the test above shows it for our implementation. Whether a given *kernel* version reproduces the reference is an empirical question per version: hence the pilot's first test.
- **At course scale.** The lab's comparison below is a toy experiment on 2.5M tokens. Whatever it shows says nothing about 1.4T-token models; it checks that the hybrid trains, gives you its cost, and gives you a first paired number with its interval.

## Lab

**Folder:** [`labs/module-05/lesson-01/`](../../labs/module-05/) · **Time:** about 90 minutes (about 40 of them unattended) · **Pass check:** `pytest labs/module-05/lesson-01` passes; `equivalence.py` shows float64 differences below $10^{-12}$ at every chunk size; your notes state the contract's decision for both hybrids with the paired intervals.

### Experiment contract

- **Question:** at equal non-embedding parameters and equal training tokens, does a 3:1 linear/full hybrid of the course model reach Baseline-0's held-out loss, and does the KDA channel-wise gate do better than Gated DeltaNet's scalar gate? Decision informed: which linear kind the 05.3 cost measurements and the module memo carry forward.
- **Hypotheses and status:** (1) chunked and recurrent forms agree to rounding — established; (2) the hybrids' held-out loss is within 0.03 nats of Baseline-0's — reported at scale (Kimi Linear, Qwen3-Next, company claims), may not appear at toy scale where 256-token contexts give linear layers no advantage; (3) KDA is at least as good as Gated DeltaNet — reported by Kimi Linear, may not appear at this scale.
- **Baseline:** `b0-s0` (toy preset, the Module 1 recipe at 256 tokens), trained in this lab with the same command shape as the arms. Tuning budget: none for any arm; all use Baseline-0's learning rate 3e-3 (say so in the limits).
- **Changed variable:** the attention kind of layers 0–2 of every 4 (KDA or Gated DeltaNet instead of GQA) and, to keep parameters equal, the SwiGLU width (384 → 308 for KDA, 315 for Gated DeltaNet at toy size). **Controlled:** Data-v0 (CPU size), tokenizer, seed 0 for weights and data order, 600 steps × 16 × 256 tokens, schedule, evaluation windows.
- **Comparison axis:** equal parameters with equal tokens. It does not answer which is cheaper to train or serve (05.3) or which is better at long context (05.3 runs Eval v1).
- **Budget:** free CPU about 35–45 minutes for three runs (measured below); main path about 1 GPU-hour, PROJECTED.
- **Metrics and decision rule:** mean held-out loss on 256 fixed validation windows of 256 tokens; paired bootstrap 95% CI of the difference against `b0-s0` (`heldout_compare.py`). Rule, stated now: "within margin" if the whole CI lies inside ±0.03 nats; "worse" if the CI lies above 0; otherwise inconclusive. Toy-scale seed noise (Module 1 project) is of the same order as the margin, so one seed can at best say "not clearly worse".
- **Correctness checks:** `pytest labs/module-05/lesson-01` and `pytest labs/common/tests/test_attention_m05.py` pass before any training.
- **Fallback evidence:** the Module 5 pilot's 30M-parameter hybrid run, labelled as provided traces.
- **Limits:** one seed, toy scale, 256-token contexts (where hybrids have nothing to gain), no tuning of the hybrids' learning rate, our PyTorch reference kernel.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 SXM or A100 80 GB; about 1 GPU-hour, PROJECTED: pilot-30m at $T = 1024$, 4,000 steps × 64 sequences = 262M tokens per arm, $321	ext{M} 	imes 262	ext{M} = 8.4 	imes 10^{16}$ training FLOPs for `b0` and $8.0 	imes 10^{16}$ for each matched hybrid (`accounting.m05_flops_per_token`); at an assumed 20% MFU on 989 TFLOP/s that is 0.12 h per arm, about 0.4 h for three plus evaluation, more if the fla kernels reach a lower MFU (record it). Not run in this build; part of the Module 5 pilot | `python labs/module-05/lesson-01/train_arms.py --variant main` (uses `--linear-mode fla`; first run `pytest labs/common/tests/test_attention_m05.py -k fla` on the GPU) |
| Free GPU (Colab/Kaggle T4) | T4, fp32, about 2 hours for three runs (PROJECTED) | `train_arms.py --variant t4` (our chunked reference on CUDA: correct, not fast) |
| Free CPU | laptop; measured below | the steps below |

### Steps

1. **Implement** `recurrent_step`, `chunk_forward`, `hybrid_pattern` and `cache_bytes` in `lab.py`; run `pytest labs/module-05/lesson-01`. The provided `LabKDA` layer uses your core; the test runs it through the correctness suite in both modes.
2. **Equivalence and memory:**

   ```bash
   python labs/module-05/lesson-01/equivalence.py
   ```

   Part 1 prints recurrent-versus-chunked differences in float64 and float32. Part 2 prints decode memory for Baseline-0's hybrid, Qwen3-Next and Kimi Linear. Check the Baseline-0 32K row by hand.
3. **Train the three arms** (unattended; rerun the same command after an interruption):

   ```bash
   python labs/module-05/lesson-01/train_arms.py
   python labs/module-05/heldout_compare.py runs/m05/l51/cpu/b0-s0 runs/m05/l51/cpu/hybrid-kda-s0 runs/m05/l51/cpu/hybrid-gdn-s0 --seq 256 --margin 0.03
   ```

4. **Report** (half a page): the equivalence table and what a float32 difference of $10^{-6}$ does and does not tell you; the parameter and FLOP columns of `heldout_compare.py` (is the comparison at equal parameters?); the paired differences and the rule's decision for each hybrid; and why the 256-token setting is the least favourable one for a hybrid.

5. **Separate the two differences** (optional, about 25 minutes): our `kda` and `gdn` kinds differ in *two* things, the decay granularity and the output-gate activation (sigmoid for KDA, as this implementation follows Kimi Linear; SiLU for Gated DeltaNet). The contract named one changed variable, so a KDA-versus-GDN difference cannot be attributed to the channel-wise gate. Run the arm that differs from `hybrid-gdn` only in the decay:

   ```bash
   python labs/module-05/lesson-01/train_arms.py --only hybrid-kda-silu
   python labs/module-05/heldout_compare.py runs/m05/l51/cpu/hybrid-gdn-s0 runs/m05/l51/cpu/hybrid-kda-silu-s0 runs/m05/l51/cpu/hybrid-kda-s0 --seq 256 --margin 0.03
   ```

Measured in this build (free CPU, Windows 11, 16-thread laptop, torch 2.14.1+cpu, fp32, toy preset at 256 tokens, 600 steps × 16 sequences = 2.46M tokens per arm, seed 0; another build job was using the CPU). Runtime: `b0` 7.3 minutes (about 5,000 tokens/s), `hybrid-kda` 22.8 minutes (about 1,800 tokens/s), `hybrid-gdn` 21 minutes (about 1,600 tokens/s): 55 minutes for the three; our chunked reference costs three to four times Baseline-0's time per token at this size. `equivalence.py`: float64 differences $\le 6 	imes 10^{-16}$ for outputs and states at every chunk size up to $T = 1{,}024$; float32 differences up to $1.4 	imes 10^{-6}$ (outputs) and $3.0 	imes 10^{-6}$ (states), growing with $T$ — rounding, the same at every chunk size, not a bug. `heldout_compare.py` (256 windows): all three arms have 0.788M non-embedding parameters (equal within 0.01%) and 2,457,600 tokens; held-out loss `b0` 5.6768, `hybrid-kda` 5.8436, difference $+0.167$, 95% CI $[+0.160, +0.173]$, **worse**; `hybrid-gdn` 5.6469, difference $-0.030$, CI $[-0.036, -0.024]$, **better** (lower loss) by the rule. So at this setting the Gated DeltaNet hybrid is slightly better than Baseline-0 and our KDA hybrid clearly worse. Step 5 (24 minutes): `hybrid-kda-silu` reaches 5.6342, $-0.043$ $[-0.049, -0.036]$ against `b0` and $-0.013$ $[-0.017, -0.008]$ against `hybrid-gdn` (within the margin), while switching the output gate back to sigmoid costs $+0.197$ $[+0.190, +0.204]$. The large KDA deficit was the output-gate activation, not the channel-wise decay: a second changed variable that the first contract did not name, found after the results and recorded as such (the template's "Changes after results"). With the decay as the only difference, channel-wise gating was slightly better than the scalar gate here. One seed, toy scale, short context, untuned learning rate: these are statements about this setting only, and the KDA result is exactly the kind of number that should trigger a check of the implementation's choices before anyone reads it as evidence about channel-wise gating.

<details>
<summary>Hint for TODO 2</summary>

Work with `b = g.cumsum(dim=2)`. `D = (b[:, :, :, None, :] - b[:, :, None, :, :])` has shape (B, H, C, C, d_k) with entry [r, i] = b_r − b_i; mask it with `torch.ones(C, C).tril()` (as bool, broadcast over the last axis) to `-inf` before `exp`. `A` and `M` are `einsum("bhrc,bhric,bhic->bhri", ...)`; zero `A` on and above the diagonal.

</details>

<details>
<summary>Hint for step 4</summary>

At 256 tokens, a full layer's attention costs about $2 \cdot 4 \cdot 32 \cdot 128$ FLOPs per token at toy width, less than the linear layer's chunked core: there is no compute to save and no cache worth shrinking. Whatever the loss difference, it says nothing about the cost case for a hybrid; it says whether the linear layers, with their short convolution and decay, model 256-token text as well as full attention at equal parameters. The settings where the cost case can be made are measured in 05.3.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-05/lesson-01/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-05/lesson-01`.

</details>

## Common mistakes

- **Testing the chunked form only with chunk = T or chunk = 1.** Both hide boundary bugs. Test several chunk sizes, a length that is not a multiple of the chunk, and a non-zero initial state.
- **Calling `exp` before masking.** Above the diagonal the exponent is positive and overflows; the masked result is then `inf · 0 = nan`.
- **Forgetting that decode carries two things.** The recurrent state *and* the short convolution's last $K - 1$ inputs; a cache without the conv tail is wrong from the first decoded token.
- **Storing the state in bf16 by default.** Check it against fp32 over long sequences before you do.
- **Quoting "75% less KV cache" for any hybrid.** The saving is at most the share of linear layers and depends on the context; Kimi Linear's 7 full layers of 27 give 74% at 1M.
- **Comparing a hybrid at 256 tokens and concluding anything about long context.** Equal-parameter loss at short context is a sanity check, not the decision.

## References

- Moonshot AI, *Kimi Linear: An Expressive, Efficient Attention Architecture*, 2025: KDA definition, chunkwise algorithm, model configuration, results. https://arxiv.org/abs/2510.26692
- S. Yang, J. Kautz, A. Hatamizadeh, *Gated Delta Networks: Improving Mamba2 with Delta Rule*, 2024. https://arxiv.org/abs/2412.06464
- S. Yang, B. Wang, Y. Zhang, Y. Shen, Y. Kim, *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*, 2024. https://arxiv.org/abs/2406.06484
- Qwen team, *Qwen3-Next-80B-A3B-Instruct* model card and `config.json` (snapshot in `labs/common/frontierlab/calc/snapshots/`). https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- Moonshot AI, *Kimi-Linear-48B-A3B-Instruct* `config.json` (checked 2026-10-04). https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct
- flash-linear-attention 0.5.2 (released 2026-07-27): `fla/ops/kda/chunk.py`, `fla/ops/gated_delta_rule/chunk.py`. https://github.com/fla-org/flash-linear-attention
- MiniMax, *Why Did MiniMax M2 End Up as a Full Attention Model?*, 2025-10-30. https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model
- Shared code: `labs/common/frontierlab/attention/deltanet.py`, `hybrid.py`, `accounting.py` (Module 5 section). Software versions: [references/versions.md](../../references/versions.md).

## Next

[05.2 · Learned sparse attention](lesson-02.md)
