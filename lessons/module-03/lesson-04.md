---
id: "03.4"
module: 3
minutes: 25
practice_minutes: 45
prerequisites: ["03.1", "03.2"]
objectives:
  - Compute how head count and head width change attention FLOPs, parameters and KV bytes for GQA and MLA, and reproduce the size of Kimi K2's 64-versus-128-head inference-FLOPs argument from its config.
  - Explain partial RoPE (Qwen3-Next, GLM-4.5, MiniMax-M2) and implement it as a rotation of a prefix of each head's channels.
  - Run a small, honestly labelled comparison of head shapes at equal attention width and state why it cannot settle the published disagreement.
volatility: implementation
sources:
  - title: "Kimi K2: Open Agentic Intelligence (section 2.3: 64 attention heads instead of 128; 0.5%-1.2% validation-loss effect; 83% more inference FLOPs at 128K)"
    url: https://arxiv.org/abs/2507.20534
  - title: "GLM-4.5 (section 2.1: 96 heads for a 5120 hidden dimension; partial RoPE; QK-Norm)"
    url: https://arxiv.org/abs/2508.06471
  - title: "Qwen3-Next-80B-A3B-Instruct config.json (head_dim 256, partial_rotary_factor 0.25)"
    url: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
  - title: "MiniMax-M2 config.json (head_dim 128, rotary_dim 64)"
    url: https://huggingface.co/MiniMaxAI/MiniMax-M2
last_verified: "2026-10-03"
---

# 03.4 · Head count and head dimension

Extension: how many attention heads a model has, and how wide each one is, looks like a detail, but it moves attention FLOPs, decode cost and what position information each head carries. Recent open models disagree in both directions: Kimi K2 halved DeepSeek-V3's 128 heads to 64 to save inference compute, GLM-4.5 uses 2.5 times more heads than its width would suggest, and Qwen3-Next uses few, very wide heads of 256 channels with RoPE on only a quarter of them. This lesson does the arithmetic behind those choices, implements partial RoPE, and runs a small comparison that shows how little a course-scale experiment can say about them.

## Why this matters at a frontier lab

Head shape is chosen once and then fixed for the model's life: checkpoints, kernels, KV-cache layouts and serving configurations all depend on it. Its costs show up in places a training-loss comparison does not see. Kimi K2's argument (section 2.3) was about inference at 128K context, a cost that equal-training-FLOPs ablations never measure. A research engineer asked "should we use 64 or 128 heads?" needs to state which costs move, by how much, and what evidence exists about quality, and to recognise that the published evidence comes from two labs with opposite conclusions under different constraints.

## The idea

### What a head costs

For a layer with $H$ query heads of query/key width $d_{qk}$ and value width $d_v$, $K$ key/value heads (GQA), and hidden width $C$:

- **projections:** $C \cdot H d_{qk}$ (queries), $C \cdot K(d_{qk} + d_v)$ (keys and values), $H d_v \cdot C$ (output) parameters, each used once per token;
- **attention scores:** $2H(d_{qk} + d_v)$ FLOPs per (query, key) pair — QKᵀ and AV;
- **KV cache:** $K(d_{qk} + d_v)$ elements per token for GQA; $d_c + d_r$ for MLA, **independent of $H$**.

Two consequences. At fixed attention width $H d$, splitting it into more, narrower heads changes neither parameters nor score FLOPs; it changes how many separate attention patterns the layer can form and how wide each dot product is. And in MLA, adding heads costs FLOPs and parameters ($q$, the per-head up-projections, $o$) but no cache: head count is a compute decision, not a memory decision.

### Kimi K2's choice

K2 keeps DeepSeek-V3's MLA dimensions but uses 64 heads instead of 128. Its report says doubling the heads "yields only modest improvements in validation loss (ranging from 0.5% to 1.2%)", while at 128K sequence length going from 64 to 128 heads (with the expert count fixed at 384) "leads to an 83% increase in inference FLOPs" (section 2.3). The decision was driven by long-context agentic inference cost (PUBLICLY DOCUMENTED, company claim for the numbers).

GLM-4.5 went the other way: "we utilize 2.5 times more attention heads (96 heads for a 5120 hidden dimension)", which "does not improve training loss compared to models with fewer heads" but "consistently improves performance on reasoning benchmarks such as MMLU and BBH" (section 2.1). Two labs, two measurements (loss versus benchmarks), two serving constraints. Neither report contradicts the other; they optimise different things.

### Partial RoPE and wide heads

RoPE rotates channel pairs at frequencies $\theta^{-2i/r}$, $i = 0 \ldots r/2 - 1$, where $r$ is the number of rotated channels. Partial RoPE rotates only the first $r < d$ channels of each head and leaves $d - r$ channels without position information, so part of each dot product compares content regardless of distance. Configs: Qwen3-Next `head_dim` 256 with `partial_rotary_factor` 0.25 ($r = 64$); GLM-4.5 `partial_rotary_factor` 0.5 with `head_dim` 128; MiniMax-M2 `rotary_dim` 64 of `head_dim` 128. MLA's decoupled key (lesson 03.1) is the same idea forced by absorption: position lives in $d_r$ channels, content in $d_n$.

Why these labs chose it is not explained in the configs; plausible reasons — cleaner long-context extrapolation because fewer channels rotate at extreme angles, and more capacity for position-independent matching — are INFERENCE/SPECULATION here, and Module 4 measures what RoPE choices do at long range.

## Worked example

### Kimi K2, 64 against 128 heads, by our arithmetic

From the bundled K2 config (61 layers, MLA with $d_n = 128$, $d_r = 64$, $d_v = 128$, $d_c = 512$, query rank 1,536, 384 experts, top-8): per (query, key) pair and layer, the naive score cost is $2H(d_n + d_r + d_v) = 640H$ FLOPs. At $S = 131{,}072$ cached tokens, for 64 heads: $640 \cdot 64 \cdot 131{,}072 \cdot 61 = 327.5$ GFLOP of attention per decoded token. `heads.py` adds the projections (which also grow with $H$) and gives:

| Context | Decode, 64 heads | Decode, 128 heads | Increase | Prefill avg, 64 | Prefill avg, 128 | Increase |
|---|---|---|---|---|---|---|
| 4K | 73.6 | 94.3 | 28.2% | 68.5 | 84.1 | 22.8% |
| 32K | 145.2 | 237.6 | 63.6% | 104.3 | 155.7 | 49.3% |
| 128K | 390.9 | 728.8 | 86.5% | 227.1 | 401.4 | 76.7% |

GFLOP per token, forward only. The report's 83% falls between our decode (86.5%) and prefill-average (76.7%) figures; it does not say which it computed or which FLOP convention it used, so our arithmetic reproduces the size of the effect, not the exact number. The KV cache is 68.6 KiB per token in both cases.

### Partial RoPE with tiny numbers

A head of width $d = 4$ with $r = 2$: the vector $(x_0, x_1, x_2, x_3)$ at position $p$ becomes $(x_0 \cos p\theta_0 - x_1 \sin p\theta_0,\; x_1 \cos p\theta_0 + x_0 \sin p\theta_0,\; x_2,\; x_3)$ in the rotate-half layout with one frequency $\theta_0 = 1$. Two tokens with identical content at positions 0 and 1 then have a dot product $x_0^2 \cos 1 + x_1^2 \cos 1 + x_2^2 + x_3^2$: the last two channels match exactly regardless of distance.

### Qwen3-Next's rotated channels

$r = 64$ channels, 32 frequencies, `rope_theta` $10^7$: wavelengths $2\pi\theta^{2i/r}$ run from 6.3 tokens to about $3.8 \times 10^7$ tokens, median about 19,900. The other 192 of 256 channels carry no position signal.

## Shapes and cost

| Arm (toy preset) | Heads × width | KV heads | Attention width | Rotated channels | KV elements per token and layer |
|---|---|---|---|---|---|
| `b0` | 4 × 32 | 2 | 128 | 32 | 128 |
| `wide-heads` | 2 × 64 | 1 | 128 | 64 | 128 |
| `partial-rope` | 2 × 64 | 1 | 128 | 16 | 128 |

All three have the same projection shapes, hence the same parameters up to the QK-norm gains (64 against 128 per layer) and the same score FLOPs: this is an equal-parameters, equal-FLOPs, equal-cache comparison by construction, which isolates head shape. Tensors are fp32 on the CPU in the free variant; main path BF16 on GPU.

`partial_rope` in the lab is the same operation as `frontierlab.attention.base.apply_rope` with a `cos` narrower than the head; `frontierlab.attention.headshape` registers `"gqa_partial"`, Baseline-0's GQA with a RoPE of width `int(rope_fraction · head_dim)` rounded to even.

## Build it

```python
from frontierlab.attention import headshape                     # registers "gqa_partial"
cfg = toy().with_(attention="gqa_partial", num_attention_heads=2, num_key_value_heads=1, head_dim=64,
                  extra={"rope_fraction": 0.25})
```

Lesson 04.2 returns to partial RoPE for long context; its `"gqa-rope-scaled"` kind (with `partial_rotary_factor` in `cfg.extra`) rotates the same channels and adds the RoPE scaling rules.

`gqa_partial` passes the same correctness suite as every Module 3 kind (op-level float64 gradient check, causal check, cached decode with chunks of 1, 3 and 7); see `labs/common/tests/test_attention_m03.py`.

## What the evidence says

- **MODEL-SPECIFIC.** 64 heads (Kimi K2), 96 heads at width 5120 (GLM-4.5), 16 heads of width 256 with 25% RoPE (Qwen3-Next) are single labs' choices, each PUBLICLY DOCUMENTED in a report section or config, with ablation evidence (where any) from the same lab.
- **The cost arithmetic is not in dispute**: our calculator reproduces the size of K2's inference-FLOPs effect from the config alone.
- **The quality effect is small and measured differently.** K2 reports 0.5–1.2% validation loss for doubling heads; GLM-4.5 reports no training-loss gain but benchmark gains for more heads. 0.5% of a loss of about 3.4 nats (a plausible value for Baseline-0 on the main path, not yet measured) is 0.017 nats — the size of the noise floor the Module 1 project measures. A course-scale experiment cannot settle this; it can show you why.

## Lab

**Folder:** [`labs/module-03/lesson-04/`](../../labs/module-03/) · **Time:** about 45 minutes · **Pass check:** `pytest labs/module-03/lesson-04` passes; your notes reproduce one row of the K2 table by hand and state what the training comparison can and cannot show.

### Experiment contract

- **Question:** at toy scale, does splitting the same attention width into fewer, wider heads, with full or 25% RoPE, change held-out loss on Data-v0 by more than evaluation noise? Decision informed: none for the architecture; this is a calibration of what a one-seed, course-scale comparison can detect.
- **Hypothesis and status:** differences below 0.02 nats; reported effects (K2, GLM-4.5) are at far larger scale and may not appear here.
- **Baseline:** `b0` (toy preset), learning rate 3e-3, not tuned for either arm (equal tuning budget: none).
- **Changed variable:** head shape (and, for `partial-rope`, the rotated fraction). **Controlled:** Data-v0, 300 steps of 16 × 128 tokens, seed 0 (same data order and initialisation scheme), the same 256 validation windows (seed 1234).
- **Comparison axis:** equal parameters, equal training FLOPs and equal cache, all by construction.
- **Budget:** free CPU, 3 runs, about 8–12 minutes; main path not needed for this extension.
- **Metrics and decision rule:** mean held-out loss over 256 fixed windows; paired bootstrap 95% interval of the per-window difference against `b0`. With one seed the interval covers evaluation noise only, not seed noise; so the rule is: report the interval and call any difference "not shown" unless it exceeds the Module 1 seed noise floor for the toy recipe (0.016 nats seed std at 600 steps, the Module 1 project's free-CPU table).
- **Correctness checks:** `gqa_partial` passes the suite (in `labs/common/tests`); `partial_rope` passes `test_lab.py`.
- **Fallback evidence:** none needed; the expected outcome is a null result.
- **Limits:** 1.8M parameters, 0.6M tokens per run, one seed, short context; says nothing about reasoning benchmarks or long-context behaviour.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | — | not needed: the arithmetic is the main result of this extension |
| Free GPU (Colab/Kaggle T4) | T4 | `heads.py --train` runs unchanged (toy preset) if you want it faster |
| Free CPU | laptop; arithmetic a few seconds, `--train` about 9.5 minutes (measured 2026-10-03, 16 threads) | `python labs/module-03/lesson-04/heads.py --train` |

### Steps

1. **Implement** `attention_flops_per_token`, `decode_attention_share` and `partial_rope` in `lab.py`; run `pytest labs/module-03/lesson-04`.
2. **Run** `python labs/module-03/lesson-04/heads.py`. Reproduce the 128K decode attention figure (327.5 GFLOP) by hand.
3. **Use your `decode_attention_share`** for Baseline-0 (one layer: $H = 12$, $d = 64$, $1{,}572{,}992 + 6{,}488{,}064 = 8.06$M matmul parameters) at 1K, 32K and 1M: at which context does attention become more than half of a layer's decode FLOPs?
4. **Train:** `python labs/module-03/lesson-04/heads.py --train`, then write the contract's conclusion.

Measured in this build (free CPU, Windows 11, 16 threads, torch 2.14.1+cpu, toy preset, 300 steps, seed 0; 9.5 minutes for the three runs): held-out loss `b0` 6.302, `wide-heads` 6.235, `partial-rope` 6.252; paired differences against `b0` over 256 windows −0.066 [−0.072, −0.061] and −0.050 [−0.055, −0.044]. Both exceed the toy seed noise floor of 0.016, so by the rule a difference is observed — for one seed, at 1.8M parameters and 300 steps, where two wider heads may simply be easier to train in so few steps. It is a reason to run more seeds, not evidence about Kimi K2's or GLM-4.5's choice, which go in opposite directions at a scale 100,000 times larger.

<details>
<summary>Hint for step 3</summary>

Attention is $2 \cdot 12 \cdot 128 \cdot S = 3{,}072 S$ FLOPs; the matmuls are $2 \times 8.06\text{M} = 16.1$M. They are equal near $S \approx 5{,}250$.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-03/lesson-04/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-03/lesson-04`.

</details>

## Common mistakes

- **Assuming more heads cost more KV cache in MLA.** They do not; they cost FLOPs and parameters. In GQA only the KV heads move the cache.
- **Comparing head counts at different attention widths** and attributing the result to the head count.
- **Reading "83% more inference FLOPs" as 83% more time.** Decode is memory-bound at small batch; FLOPs translate into time only where compute binds (large batch, prefill).
- **Treating two labs' opposite choices as a contradiction.** They measured different outcomes under different constraints.

## References

- Moonshot AI, *Kimi K2*, section 2.3. https://arxiv.org/abs/2507.20534
- GLM-4.5 Team (Zhipu AI), *GLM-4.5*, section 2.1. https://arxiv.org/abs/2508.06471
- Qwen, Qwen3-Next-80B-A3B-Instruct `config.json` (snapshot in `labs/common/frontierlab/calc/snapshots/`, fetched 2026-10-03). https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- MiniMax, MiniMax-M2 `config.json` (snapshot, fetched 2026-10-03). https://huggingface.co/MiniMaxAI/MiniMax-M2

## Next

The module project: [Attention memo — GQA, MLA and local/global against Baseline-0](../../projects/module-03-attention-memo.md). Module 4 then asks whether the model actually uses its context.
